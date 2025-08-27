"""
Analyze common high-error molecules across multiple prediction CSVs (ECFP / GLEM / FUSED)
+ Optional 3D/ionization/solubility proxies & OOD (nearest-neighbor Tanimoto) analysis
======================================================================================

This script:
  1) Loads three prediction CSVs (must contain at least: id, abs_residual). If needed,
     it will compute abs_residual from residual or true/pred and will also compute
     rank_by_abs_error if not present.
  2) Picks top-K highest-error IDs from each model and takes the INTERSECTION
     (or majority vote) as "common outliers".
  3) Builds a base table of IDs with a 'group' column: {common, background}.
  4) Reads SMILES from one or more .smi files and computes RDKit features / substructure flags /
     Murcko scaffold. **Optionally adds 3D shape descriptors (PMI/NPR/Rgyr/etc.) and
     acid/base site counts + ESOL logS proxy.** Joins features via left-merge so 'group'
     always exists.
  5) Compares common vs background:
       - Continuous features: Mann–Whitney U test + Cliff's delta
       - Binary substructures: Fisher's exact test + odds ratio
  6) (Optional) OOD: for each ID, compute ECFP nearest-neighbor Tanimoto vs TRAIN set.
  7) Saves results in OUT_DIR.

Example
-------
python AnalyzingOutliers.py \
  --ecfp ./REALEND/ecfp_test.csv \
  --glem ./REALEND/glem_test.csv \
  --fused ./REALEND/fused_test.csv \
  --smi ./Embedding/A_glem.smi ./Embedding/O_glem.smi \
  --out ./REALEND \
  --topk 50 \
  --majority 2 \
  --compute-3d \
  --ood-train-smi ./Embedding/A_glem.smi \
  --ood-threshold 0.35

CSV expectations
----------------
- Columns: id, abs_residual  (optionally true, pred, residual, rank_by_abs_error)
- If rank_by_abs_error missing, we compute it from abs_residual (largest error = rank 1)
- If abs_residual missing, we compute from residual or |true - pred|

"""
from __future__ import annotations
import argparse
import os
import re
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

# RDKit & SciPy
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS, Draw, rdMolDescriptors
from rdkit import DataStructs
from scipy.stats import mannwhitneyu, fisher_exact

# ------------------------------
# Helpers: column normalization / ID coercion
# ------------------------------

def _strip_bom_ws(s: str) -> str:
    return str(s).strip().replace("﻿", "")


def _normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _strip_bom_ws(x).lower())


def coerce_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is a column literally named 'id'. If not, try to find/rename one.
    If index looks like id, reset it. As a last resort, create from index.
    """
    df = df.copy()
    df.columns = [_strip_bom_ws(c) for c in df.columns]

    if "id" not in df.columns:
        # case-insensitive 'id'
        for c in list(df.columns):
            if _strip_bom_ws(c).lower() == "id":
                df = df.rename(columns={c: "id"})
                break

    if "id" not in df.columns:
        # index named id
        if df.index.name and _strip_bom_ws(df.index.name).lower() == "id":
            df = df.reset_index().rename(columns={df.index.name: "id"})

    if "id" not in df.columns:
        # id-like column
        norms = {c: _normalize_name(c) for c in df.columns}
        PRIMARY = {"id", "molid", "moleculeid", "compoundid"}
        hit = next((c for c, n in norms.items() if n in PRIMARY), None)
        if hit is None:
            hit = next((c for c, n in norms.items() if n.endswith("id") or "id" in n), None)
        if hit is not None:
            df = df.rename(columns={hit: "id"})

    if "id" not in df.columns:
        # last resort
        df = df.reset_index().rename(columns={"index": "id"})

    df["id"] = df["id"].astype(str).str.strip()
    return df


# ------------------------------
# Load prediction CSVs robustly
# ------------------------------

def ensure_abs_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure abs_residual
    cols_lower = {c.lower(): c for c in df.columns}
    if "abs_residual" not in cols_lower:
        if "residual" in cols_lower:
            c = cols_lower["residual"]
            df["abs_residual"] = df[c].astype(float).abs()
        elif "true" in cols_lower and "pred" in cols_lower:
            t = cols_lower["true"]; p = cols_lower["pred"]
            df["abs_residual"] = (df[t].astype(float) - df[p].astype(float)).abs()
        else:
            raise KeyError("abs_residual not found and cannot be derived (need residual or true/pred)")

    # keep highest-error row per id
    df = df.sort_values("abs_residual", ascending=False).groupby("id", as_index=False).first()

    # ensure rank_by_abs_error
    if "rank_by_abs_error" not in df.columns:
        df["rank_by_abs_error"] = (-df["abs_residual"]).rank(method="min").astype(int)

    return df


def load_preds(path: str, tag: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = coerce_id_column(df)
    df = ensure_abs_and_rank(df)
    # Keep compact columns (keep true/pred if present)
    cols = ["id", "abs_residual", "rank_by_abs_error"]
    for c in ("true", "pred"):
        if c in {x.lower() for x in df.columns}:
            real = next(x for x in df.columns if x.lower() == c)
            cols.append(real)
    df = df[[c for c in cols if c in df.columns]].copy()
    # tag prefix
    ren = {"abs_residual": f"abs_{tag}", "rank_by_abs_error": f"rank_{tag}"}
    for c in ("true","pred"):
        real = next((x for x in df.columns if x.lower() == c), None)
        if real: ren[real] = f"{c}_{tag}"
    df = df.rename(columns=ren)
    return df


# ------------------------------
# SMILES mapping
# ------------------------------

def read_smi(paths: List[str], allowed_ids: Set[str] | None = None) -> Dict[str, str]:
    """Reads one or more .smi files.
    Format: first token = SMILES; one of the remaining tokens is the ID.
    If allowed_ids is given, choose the first token that matches it.
    Otherwise, choose the last token, or the longest alphanumeric token.
    """
    mp: Dict[str, str] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                toks = line.split()
                smi = toks[0]
                _id = None
                if allowed_ids:
                    for t in toks[1:]:
                        if t in allowed_ids:
                            _id = t; break
                if _id is None:
                    cand = [t for t in toks[1:] if any(ch.isalnum() for ch in t)]
                    if cand:
                        _id = cand[-1]
                if _id:
                    mp[str(_id)] = smi
    return mp


def to_mol(_id: str, id2smi: Dict[str, str]):
    smi = id2smi.get(str(_id))
    if not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


# ------------------------------
# Feature computation (2D + optional 3D + acid/base + ESOL proxy)
# ------------------------------
SMARTS = {
    "nitro":        "[N+](=O)[O-]",
    "sulfonamide":  "S(=O)(=O)N",
    "sulfone":      "S(=O)(=O)[!O]",
    "amide":        "C(=O)N",
    "ester":        "C(=O)O",
    "carboxyl":     "C(=O)[O-,OH]",
    "tertiary_amine":"[NX3;H0;!$(NC=O)]",
    "quat_ammonium":"[NX4+]",
    "phenyl":       "c1ccccc1",
    "hetero_5ring": "[r5;!$([r5]~[r5])]",
    "thiophene":    "s1cccc1",
    "indole":       "c1ccc2[nH]ccc2c1",
    "halogen":      "[F,Cl,Br,I]",
    # Acid / base site proxies
    "acid_carboxyl": "[CX3](=O)[OX2H1,OX1H0-]",
    "acid_sulfonic": "S(=O)(=O)[OX2H,OX1-]",
    "base_amine":    "[NX3;H2,H1,H0;!$(NC=O)]",
}
SMARTS_PATT = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}


def _esol_logS_proxy(mw, logp, rotbonds, arom_rings):
    # Delaney-like linear proxy (approximate)
    # logS ≈ 0.16 - 0.63*logP - 0.0062*MW + 0.066*RotBonds - 0.74*AromaticRings
    return 0.16 - 0.63*logp - 0.0062*mw + 0.066*rotbonds - 0.74*arom_rings


def generate_3d_conformer(mol: Chem.Mol, max_iters: int = 200) -> Chem.Mol | None:
    try:
        mH = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        if AllChem.EmbedMolecule(mH, params=params) != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mH, maxIters=max_iters)
        except Exception:
            pass
        m = Chem.RemoveHs(mH)
        return m if m.GetNumConformers() > 0 else None
    except Exception:
        return None


def compute_features_2d(mol) -> Dict[str, float] | None:
    if mol is None:
        return None
    feats = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FracCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "HeteroAtoms": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
        "HalogenCount": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53)),
    }
    for name, patt in SMARTS_PATT.items():
        feats[f"has_{name}"] = int(mol.HasSubstructMatch(patt))
    try:
        feats["scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        feats["scaffold"] = ""
    feats["acid_sites"] = int(mol.HasSubstructMatch(SMARTS_PATT["acid_carboxyl"])) + int(mol.HasSubstructMatch(SMARTS_PATT["acid_sulfonic"]))
    feats["basic_sites"] = len(mol.GetSubstructMatches(SMARTS_PATT["base_amine"]))
    # ESOL proxy
    feats["logS_esol_proxy"] = _esol_logS_proxy(feats["MW"], feats["LogP"], feats["RotBonds"], feats["AromRings"])
    return feats


def compute_features_3d(mol) -> Dict[str, float] | None:
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        return None
    f = {
        "PMI1": rdMolDescriptors.CalcPMI1(mol),
        "PMI2": rdMolDescriptors.CalcPMI2(mol),
        "PMI3": rdMolDescriptors.CalcPMI3(mol),
        "NPR1": rdMolDescriptors.CalcNPR1(mol),
        "NPR2": rdMolDescriptors.CalcNPR2(mol),
        "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration(mol),
        "Asphericity": rdMolDescriptors.CalcAsphericity(mol),
        "Eccentricity": rdMolDescriptors.CalcEccentricity(mol),
        "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor(mol),
        "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol),
    }
    # Derived ratios
    if f["PMI3"] > 1e-9:
        f["PMI1_over_PMI3"] = f["PMI1"]/f["PMI3"]
        f["PMI2_over_PMI3"] = f["PMI2"]/f["PMI3"]
    return f


# ------------------------------
# Statistics
# ------------------------------

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = 0; lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    n = a.size * b.size
    return float((gt - lt) / n) if n else float("nan")


# ------------------------------
# OOD (nearest neighbor Tanimoto to training set)
# ------------------------------

def morgan_fp_bits(mol, radius=2, nBits=2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)


def ood_nn_tanimoto(out_ids: List[str], train_ids: List[str], id2smi: Dict[str,str],
                    radius=2, nBits=2048) -> pd.DataFrame:
    # Build fps for training
    train_mols = [to_mol(i, id2smi) for i in train_ids]
    train_mols = [m for m in train_mols if m is not None]
    fps_train = [morgan_fp_bits(m, radius, nBits) for m in train_mols]

    rows = []
    for _id in out_ids:
        m = to_mol(_id, id2smi)
        if m is None:
            rows.append({"id": _id, "nn_tanimoto": np.nan}); continue
        fp = morgan_fp_bits(m, radius, nBits)
        if len(fps_train) == 0:
            rows.append({"id": _id, "nn_tanimoto": np.nan}); continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
        rows.append({"id": _id, "nn_tanimoto": float(np.max(sims)) if len(sims)>0 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------
# Main flow
# ------------------------------

def pick_common_ids(df_e: pd.DataFrame, df_g: pd.DataFrame, df_f: pd.DataFrame, topk: int, majority: int) -> Tuple[Set[str], Set[str]]:
    sets = []
    sets.append(set(df_e.nsmallest(topk, "rank_ecfp")["id"]) if "rank_ecfp" in df_e.columns else set(df_e.nlargest(topk, "abs_ecfp")["id"]))
    sets.append(set(df_g.nsmallest(topk, "rank_glem")["id"]) if "rank_glem" in df_g.columns else set(df_g.nlargest(topk, "abs_glem")["id"]))
    sets.append(set(df_f.nsmallest(topk, "rank_fused")["id"]) if "rank_fused" in df_f.columns else set(df_f.nlargest(topk, "abs_fused")["id"]))
    from collections import Counter
    cnt = Counter(i for s in sets for i in s)
    common = {i for i,c in cnt.items() if c >= max(1, majority)}
    all_ids = set(pd.concat([df_e["id"], df_g["id"], df_f["id"]], ignore_index=True).astype(str))
    background = all_ids - common
    return common, background


def analyze(ecfp_csv: str, glem_csv: str, fused_csv: str, smi_files: List[str], out_dir: str,
            topk: int = 50, majority: int = 3, bg_sample: int = 5000, mcs_timeout: int = 10,
            compute_3d: bool = False, ood_train_smi: List[str] | None = None, ood_threshold: float = 0.35,
            ecfp_radius: int = 2, ecfp_bits: int = 2048) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load predictions
    ecfp  = load_preds(ecfp_csv,  tag="ecfp")
    glem  = load_preds(glem_csv,  tag="glem")
    fused = load_preds(fused_csv, tag="fused")

    # 2) Determine common & background ids (majority vote allowed)
    common_ids, bg_ids = pick_common_ids(ecfp, glem, fused, topk=topk, majority=majority)
    print(f"#common_outliers={len(common_ids)}  #background={len(bg_ids)} (topk={topk}, majority={majority})")

    # Save per-id error summary for common ids
    merged = ecfp.merge(glem, on="id", how="outer").merge(fused, on="id", how="outer")
    merged["mean_abs_residual"] = merged[[c for c in merged.columns if c.startswith("abs_")]].mean(axis=1)
    merged["avg_rank"] = merged[[c for c in merged.columns if c.startswith("rank_")]].mean(axis=1)
    merged_common = merged[merged["id"].isin(common_ids)].copy().sort_values(["avg_rank", "mean_abs_residual"], ascending=[True, False])
    merged_common.to_csv(os.path.join(out_dir, "common_ids.csv"), index=False)

    # 3) Base id→group table
    base = pd.DataFrame({"id": list(common_ids) + list(bg_ids)})
    base = coerce_id_column(base)
    membership = {str(i): "common" for i in common_ids}
    membership.update({str(i): "background" for i in bg_ids})
    base["group"] = base["id"].map(membership).fillna("unknown")

    # 4) SMILES & features
    all_ids_for_match = set(pd.concat([ecfp["id"], glem["id"], fused["id"]], ignore_index=True).astype(str))
    id2smi = read_smi(smi_files, allowed_ids=all_ids_for_match)

    feat_rows: List[Dict] = []
    # Common (all)
    for _id in common_ids:
        m = to_mol(_id, id2smi)
        f2 = compute_features_2d(m)
        if f2:
            row = {"id": str(_id), **f2}
            if compute_3d:
                m3 = generate_3d_conformer(m)
                f3 = compute_features_3d(m3)
                if f3: row.update({f"3D_{k}": v for k, v in f3.items()})
            feat_rows.append(row)
    # Background sample
    for _id in list(bg_ids)[: int(bg_sample) ]:
        m = to_mol(_id, id2smi)
        f2 = compute_features_2d(m)
        if f2:
            row = {"id": str(_id), **f2}
            if compute_3d:
                m3 = generate_3d_conformer(m)
                f3 = compute_features_3d(m3)
                if f3: row.update({f"3D_{k}": v for k, v in f3.items()})
            feat_rows.append(row)

    df_feat = pd.DataFrame(feat_rows) if feat_rows else pd.DataFrame(columns=["id"])  # may be empty
    df = base.merge(df_feat, on="id", how="left")

    # 5) Stats
    cont_cols = [c for c in [
        "MW","LogP","TPSA","HBD","HBA","RotBonds","RingCount","AromRings","FracCSP3","FormalCharge","HeteroAtoms","HalogenCount",
        "acid_sites","basic_sites","logS_esol_proxy"
    ] + [c for c in df.columns if c.startswith("3D_")] if c in df.columns]

    cont_res = []
    for c in cont_cols:
        ca = df.loc[df["group"].eq("common"), c].dropna().values
        cb = df.loc[df["group"].eq("background"), c].dropna().values
        if len(ca) >= 3 and len(cb) >= 3:
            stat, p = mannwhitneyu(ca, cb, alternative="two-sided")
            d = cliffs_delta(ca, cb)
            cont_res.append({
                "feature": c,
                "common_med": float(np.median(ca)),
                "bg_med": float(np.median(cb)),
                "p_value": float(p),
                "cliffs_delta": float(d),
                "n_common": int(len(ca)),
                "n_bg": int(len(cb)),
            })
    cont_tbl = pd.DataFrame(cont_res).sort_values(["p_value","cliffs_delta"], ascending=[True, False]) if cont_res else pd.DataFrame()

    bin_cols = [c for c in df.columns if c.startswith("has_")]
    bin_res = []
    mask_common = df["group"].eq("common")
    mask_bg     = df["group"].eq("background")
    for c in bin_cols:
        pos = df[c].fillna(0).astype(int).eq(1)
        neg = df[c].fillna(0).astype(int).eq(0)
        a1 = int((mask_common & pos).sum()); a0 = int((mask_common & neg).sum())
        b1 = int((mask_bg & pos).sum());     b0 = int((mask_bg & neg).sum())
        table = [[a1, a0], [b1, b0]]
        try:
            OR, p = fisher_exact(table)
        except Exception:
            OR, p = (float("nan"), 1.0)
        bin_res.append({
            "substructure": c.replace("has_",""),
            "common_pos": a1, "bg_pos": b1,
            "odds_ratio": float(OR), "p_value": float(p),
        })
    bin_tbl = pd.DataFrame(bin_res).sort_values(["p_value","odds_ratio"], ascending=[True, False]) if bin_res else pd.DataFrame()

    # 6) MCS (unchanged; may be empty)
    mcs_smarts = ""
    mols_common = [to_mol(i, id2smi) for i in common_ids]
    mols_common = [m for m in mols_common if m is not None]
    if len(mols_common) >= 2:
        try:
            mcs = rdFMCS.FindMCS(mols_common, timeout=int(mcs_timeout), ringMatchesRingOnly=False, completeRingsOnly=False)
            if mcs.numAtoms > 0:
                mcs_smarts = mcs.smartsString
        except Exception:
            mcs_smarts = ""

    # 7) OOD (optional)
    if ood_train_smi:
        id2smi_train = read_smi(ood_train_smi)
        # derive train id list from files
        train_ids = list(id2smi_train.keys())
        # Common & (optionally) background sample
        ood_common = ood_nn_tanimoto(list(common_ids), train_ids, {**id2smi_train, **id2smi}, radius=ecfp_radius, nBits=ecfp_bits)
        ood_common["ood_flag"] = (ood_common["nn_tanimoto"].fillna(0) < float(ood_threshold)).astype(int)
        ood_common.to_csv(os.path.join(out_dir, "ood_nn_tanimoto_common.csv"), index=False)
        # Background sample for comparison
        bg_sample_ids = list(bg_ids)[: int(bg_sample) ]
        ood_bg = ood_nn_tanimoto(bg_sample_ids, train_ids, {**id2smi_train, **id2smi}, radius=ecfp_radius, nBits=ecfp_bits)
        ood_bg.to_csv(os.path.join(out_dir, "ood_nn_tanimoto_background.csv"), index=False)

    # 8) Save
    cont_path = os.path.join(out_dir, "common_vs_bg_continuous.csv")
    bin_path  = os.path.join(out_dir, "common_vs_bg_substructures.csv")
    base_path = os.path.join(out_dir, "grouped_ids.csv")
    mcs_path  = os.path.join(out_dir, "common_mcs.smarts")

    base.to_csv(base_path, index=False)
    if not cont_tbl.empty:
        cont_tbl.to_csv(cont_path, index=False)
    if not bin_tbl.empty:
        bin_tbl.to_csv(bin_path, index=False)
    with open(mcs_path, "w") as f:
        f.write(mcs_smarts)

    print("Saved:")
    print("  ", base_path)
    print("  ", cont_path if os.path.exists(cont_path) else "(no continuous feature stats — insufficient data)")
    print("  ", bin_path if os.path.exists(bin_path) else "(no substructure stats — insufficient data)")
    print("  ", mcs_path, ("(empty)" if not mcs_smarts else ""))
    if ood_train_smi:
        print("  ", os.path.join(out_dir, "ood_nn_tanimoto_common.csv"))
        print("  ", os.path.join(out_dir, "ood_nn_tanimoto_background.csv"))


def parse_args():
    ap = argparse.ArgumentParser(description="Analyze common high-error molecules across models")
    ap.add_argument("--ecfp", required=True, help="ECFP predictions CSV")
    ap.add_argument("--glem", required=True, help="GLEM predictions CSV")
    ap.add_argument("--fused", required=True, help="FUSED predictions CSV")
    ap.add_argument("--smi", nargs="+", required=True, help="One or more .smi files (SMILES first token, then ID tokens)")
    ap.add_argument("--out", default="./REALEND", help="Output directory")
    ap.add_argument("--topk", type=int, default=50, help="Per-model top-K by abs error (or rank) to select")
    ap.add_argument("--majority", type=int, default=3, help="How many models must include an ID (1..3). Use 2 for majority.")
    ap.add_argument("--bg-sample", type=int, default=5000, help="Background sampling size")
    ap.add_argument("--mcs-timeout", type=int, default=10, help="MCS timeout seconds")
    ap.add_argument("--compute-3d", action="store_true", help="Compute 3D shape descriptors (ETKDG+UFF)")
    ap.add_argument("--ood-train-smi", nargs="+", help=".smi file(s) that define the TRAIN set for OOD Tanimoto analysis")
    ap.add_argument("--ood-threshold", type=float, default=0.35, help="OOD flag threshold on NN Tanimoto (ECFP)")
    ap.add_argument("--ecfp-radius", type=int, default=2, help="ECFP radius for OOD analysis")
    ap.add_argument("--ecfp-bits", type=int, default=2048, help="ECFP bits for OOD analysis")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(
        ecfp_csv=args.ecfp,
        glem_csv=args.glem,
        fused_csv=args.fused,
        smi_files=args.smi,
        out_dir=args.out,
        topk=args.topk,
        majority=args.majority,
        bg_sample=args.bg_sample,
        mcs_timeout=args.mcs_timeout,
        compute_3d=args.compute_3d,
        ood_train_smi=args.ood_train_smi,
        ood_threshold=args.ood_threshold,
        ecfp_radius=args.ecfp_radius,
        ecfp_bits=args.ecfp_bits,
    )
