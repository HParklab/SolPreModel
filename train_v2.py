"""
train_v2_new_modified.py

Goal
====
- Fair, apples-to-apples comparison pipeline that applies **the same downstream steps**
  (input normalization → scalar regressor → MSE loss → metrics → plotting)
  to multiple embedding types (e.g., **ECFP** and **GLEM**).
- Saves per-model predictions/plots, plus an overlay scatter and a summary CSV.

Assumptions
===========
- You have embeddings stored in NPZ as dicts under key "emb": { id: np.ndarray }.
- You have label files (.smi-like) whose line contains the ID token and ends with a float label.
- You have a split JSON with keys "train"|"valid"|"test" listing IDs.
- You have Projection.py with EmbeddingProjector supporting (emb_type, method, sequence_mode) and .fit().

What this script does
=====================
1) Loads embeddings + labels and intersects IDs with a split file.
2) For each embedding choice (ECFP, GLEM), runs the exact same steps:
   - Project to (B, 128) using EmbeddingProjector with sequence_mode=False (fair comparison)
   - Fit InputNorm on A-train fused/projected features only
   - Train scalar regressor with AdamW + ReduceLROnPlateau + early stopping
   - Evaluate on test set, save predictions CSV & plots
3) Writes a summary CSV across runs and an overlay scatter (ECFP vs GLEM on one plot)

Notes
=====
- For E2E sequence models, set sequence_mode=True in your other scripts.
- Guarded scaler usage is recommended in Projection.py.

"""
from __future__ import annotations
import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Helvetica Neue"

# -----------------------------
# Config (edit these paths)
# -----------------------------
OUT_DIR = "./EGNN"
SEED = 42
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 20
DROPOUT = 0.3
HIDDEN = 128
PROJ_OUT_DIM = 128  # unified dim for fair comparison

# Fill in your actual paths below.
# A_* are train/valid domain, O_* are test domain.
CFG = {
    "ECFP": {
        "emb_A": "./Embedding/A_egnn.emb.npz",   # e.g., A_ecfp4_fresh.npz (key='emb' with dict{id: vec})
        "smi_A": "./Embedding/A_glem.smi",      # file containing IDs and last float = label
        "emb_O": "./Embedding/O_egnn.emb.npz",
        "smi_O": "./Embedding/O_glem.smi",
        "split": "./DataSplit/split.v2.json",
        "emb_type": "ecfp",
        "method": "umap",               # pca|umap|ae|linear
    },
    "GLEM": {
        "emb_A": "./Embedding/A_egnn.emb.npz",
        "smi_A": "./Embedding/A_glem.smi",
        "emb_O": "./Embedding/O_egnn.emb.npz",
        "smi_O": "./Embedding/O_glem.smi",
        "split": "./Datasplit/split.v2.json",
        "emb_type": "glem",
        "method": "linear",            # GLEM usually already 128 → just linear/identity
    },
}
TO_RUN = ["ECFP"]

# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed:int=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Data utils
# -----------------------------
def load_npz_dict(path: str, key: str = "emb") -> Dict[str, np.ndarray]:
    """
    Robust loader for NPZ embeddings.

    Supports:
    A) np.savez(..., emb=dict)                      -> 0-d object array, .item() works
    B) np.savez(..., emb=matrix, ids=list/array)    -> 2D emb + 1D ids, zipped to dict
    C) np.savez(..., keys=..., vals=...)            -> zipped to dict
    D) np.savez(..., emb=obj_array_of_dicts)        -> 1D object array whose elems are dicts
    E) np.savez(..., **{id: vector})                -> each npz field is an id (rare)
    """
    npz = np.load(path, allow_pickle=True)
    files = list(npz.files)

    # --- Preferred key path ---
    if key in files:
        arr = npz[key]

        # A) 0-d object -> dict
        if arr.dtype == object and arr.ndim == 0:
            obj = arr.item()
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{key} is object but not a dict")
            return obj

        # D) 1-d object array whose elements are dicts -> merge
        if arr.dtype == object and arr.ndim == 1 and len(arr) > 0 and isinstance(arr[0], dict):
            out: Dict[str, np.ndarray] = {}
            for d in arr:
                out.update(d)
            return out

        # B) matrix + side 'ids'
        if arr.ndim == 2 and ("ids" in files or "keys" in files or "id" in files):
            ids_name = "ids" if "ids" in files else ("keys" if "keys" in files else "id")
            ids = npz[ids_name]
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            if len(ids) != arr.shape[0]:
                raise ValueError(f"{path}: len(ids)={len(ids)} != emb.rows={arr.shape[0]}")
            return {str(i): arr[idx] for idx, i in enumerate(ids)}

        # Fallback: if it's 2D but no ids, synthesize integer IDs
        if arr.ndim == 2:
            return {str(i): arr[i] for i in range(arr.shape[0])}

        raise ValueError(f"{path}:{key} has unexpected shape/dtype: shape={arr.shape}, dtype={arr.dtype}")

    # --- Alternate layouts ---
    # C) keys + vals
    if ("keys" in files) and ("vals" in files):
        keys = npz["keys"]
        vals = npz["vals"]
        keys = keys.tolist() if hasattr(keys, "tolist") else list(keys)
        if isinstance(keys[0], bytes):  # sometimes saved as bytes
            keys = [k.decode("utf-8") for k in keys]
        if vals.ndim == 1 and vals.dtype == object:
            # vals = object array of vectors
            return {k: vals[i] for i, k in enumerate(keys)}
        elif vals.ndim == 2:
            return {k: vals[i] for i, k in enumerate(keys)}
        else:
            raise ValueError(f"{path}: unsupported vals shape {vals.shape}")

    # E) each field is an id
    if all(npz[f].ndim in (1, 2) for f in files):
        # try to treat each field name as an ID
        try:
            return {f: npz[f] for f in files}
        except Exception as e:
            raise ValueError(f"{path}: couldn't build dict from fields {files}: {e}")

    raise ValueError(f"{path}: none of the supported formats matched; fields={files}")


def parse_labels_by_ids(label_path: str, id_candidates: set[str]) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # comments
                continue
            toks = line.split()
            # label is the last float token
            try:
                y = float(toks[-1])
            except Exception:
                continue
            # pick the first token in the line that exactly matches an ID candidate
            id_match = None
            for t in toks:
                if t in id_candidates:
                    id_match = t
                    break
            if id_match is None:
                continue
            vals[id_match] = y
    return vals


def deflatten_to_TD(x: np.ndarray, bases=(128, 256)) -> np.ndarray:
    """If x is 1D, try to reshape into (T, D) with D in bases. Else return as 2D.
    If none fits, return shape (1, D)."""
    if x.ndim == 2:
        return x
    if x.ndim != 1:
        # already something else (e.g., 3D) → trust caller
        return x
    n = x.shape[0]
    for D in bases:
        if n % D == 0:
            T = n // D
            return x.reshape(T, D)
    return x.reshape(1, n)

def _print_one_line_metrics(tag, metrics, n):
    print(f"[{tag}] n={n}  RMSE={metrics['RMSE']:.6f}  MAE={metrics['MAE']:.6f}  R2={metrics['R2']:.6f}  Pearson={metrics['Pearson']:.6f}")

def save_ranked_preds_csv(path, ids, y_true, y_pred):
    import numpy as np, pandas as pd
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_pred - y_true
    abs_residual = np.abs(residual)
    # rank: 1 = 가장 큰 오차
    order = np.argsort(-abs_residual)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(abs_residual) + 1)

    df = pd.DataFrame({
        "id": ids,
        "true": y_true,
        "pred": y_pred,
        "residual": residual,
        "abs_residual": abs_residual,
        "rank_by_abs_error": ranks,
    })
    df.to_csv(path, index=False)

# -----------------------------
# Dataset & Collate
# -----------------------------
class PairDataset(Dataset):
    def __init__(self, emb_dict: Dict[str, np.ndarray], labels: Dict[str, float], ids: List[str]):
        self.ids = ids
        self.emb = emb_dict
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        k = self.ids[i]
        x = self.emb[k]
        # enforce (T,D)
        if x.ndim == 1:
            x = deflatten_to_TD(x)
        y = float(self.labels[k])
        return x.astype(np.float32), y, k


def pad_collate(batch):
    xs, ys, ks = zip(*batch)
    # xs are (T,D) variable T
    Ts = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    B = len(xs)
    maxT = max(Ts)
    X = np.zeros((B, maxT, D), dtype=np.float32)
    M = np.zeros((B, maxT), dtype=np.bool_)
    for i, x in enumerate(xs):
        t = x.shape[0]
        X[i, :t, :] = x
        M[i, :t] = True
    X = torch.from_numpy(X)
    M = torch.from_numpy(M)
    Y = torch.tensor(ys, dtype=torch.float32)
    return X, M, Y, ks


# -----------------------------
# Model bits
# -----------------------------
class InputNorm(nn.Module):
    """Channel-wise normalization for (B,D) vectors.
    Use .fit(loader, projector) to compute mean/std on TRAIN features only.
    """
    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))

    @torch.no_grad()
    def fit(self, loader: DataLoader, projector: nn.Module, device: torch.device):
        feats = []
        for X, M, Y, _ in loader:
            X = X.to(device)
            # sequence_mode=False → projector outputs (B, D)
            Z = projector(X)  # (B, D)
            feats.append(Z.detach().cpu())
        if len(feats) == 0:
            return
        Fcat = torch.cat(feats, dim=0)
        m = Fcat.mean(dim=0)
        s = Fcat.std(dim=0) + 1e-6
        self.mean.copy_(m)
        self.std.copy_(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class Regressor(nn.Module):
    def __init__(self, d_in: int, mid: int = HIDDEN, dropout: float = DROPOUT, nlayers: int = 2):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(nlayers - 1):
            layers += [nn.Linear(d, mid), nn.ReLU(), nn.Dropout(dropout)]
            d = mid
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------
# Projection wrapper (uses Projection.py)
# -----------------------------
from Projection import EmbeddingProjector  # must be available in PYTHONPATH


def build_projector(in_dim: int, emb_type: str, method: str, out_dim: int = PROJ_OUT_DIM):
    # For fair comparison we summarize sequences → (B,D), so sequence_mode=False here
    return EmbeddingProjector(
        in_dim=in_dim,
        out_dim=out_dim,
        emb_type=emb_type,
        method=method,
        sequence_mode=False,
    )


# -----------------------------
# Training / Eval
# -----------------------------
@torch.no_grad()
def collect_features(loader: DataLoader, projector: nn.Module, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    for X, M, Y, _ in loader:
        X = X.to(device)
        Z = projector(X)
        feats.append(Z.detach().cpu())
        labels.append(Y)
    if len(feats) == 0:
        return torch.empty(0, PROJ_OUT_DIM), torch.empty(0)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def train_one(name: str,
              tr_loader: DataLoader,
              va_loader: DataLoader,
              projector: nn.Module,
              inorm: InputNorm,
              device: torch.device) -> Tuple[nn.Module, Dict[str, float], str]:
    model = Regressor(d_in=PROJ_OUT_DIM).to(device)

    # Set up optim & scheduler
    wd_exclude = []
    wd_include = []
    for n, p in model.named_parameters():
        if p.ndimension() == 1 or n.endswith("bias"):
            wd_exclude.append(p)
        else:
            wd_include.append(p)
    optim = torch.optim.AdamW([
        {"params": wd_include, "weight_decay": WEIGHT_DECAY},
        {"params": wd_exclude, "weight_decay": 0.0},
    ], lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.2, patience=3)

    best = {"val": math.inf}
    best_path = os.path.join(OUT_DIR, f"(2){name}_best.pt")
    no_improve = 0
    history = {"tr": [], "va": []}

    for epoch in range(9999):
        # --- train ---
        model.train()
        tr_loss = 0.0
        n_samples = 0
        for X, M, Y, _ in tr_loader:
            X = X.to(device)
            Y = Y.to(device)
            # features
            with torch.no_grad():
                Z = projector(X)          # (B,D)
                Z = inorm(Z)              # (B,D)
            pred = model(Z)               # (B,)
            loss = F.mse_loss(pred, Y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            bs = Y.shape[0]
            tr_loss += loss.item() * bs
            n_samples += bs
        tr_loss /= max(1, n_samples)
        history["tr"].append(tr_loss)

        # --- valid ---
        model.eval()
        va_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for X, M, Y, _ in va_loader:
                X = X.to(device)
                Y = Y.to(device)
                Z = projector(X)
                Z = inorm(Z)
                pred = model(Z)
                loss = F.mse_loss(pred, Y)
                bs = Y.shape[0]
                va_loss += loss.item() * bs
                n_samples += bs
        va_loss /= max(1, n_samples)
        history["va"].append(va_loss)

        scheduler.step(va_loss)
        print(f"[LR] {scheduler.get_last_lr()}")


        print(f"[Epoch {epoch:03d}] tr={tr_loss:.6f}  va={va_loss:.6f}")

        if va_loss + 1e-12 < best["val"]:
            best["val"] = va_loss
            torch.save({"model": model.state_dict()}, best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopped.")
                break

    # Plot loss curves
    curve_path = os.path.join(OUT_DIR, f"(2)loss_v2_{name}.png")
    plt.figure()
    plt.plot(history["tr"], label="train")
    plt.plot(history["va"], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Loss Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()

    return model, {"best_val_loss": float(best["val"])}, curve_path


@torch.no_grad()
def eval_metrics(model: nn.Module, projector: nn.Module, inorm: InputNorm, loader: DataLoader, device: torch.device):
    model.eval()
    ys, ps, ids = [], [], []
    for X, M, Y, ks in loader:
        X = X.to(device)
        Y = Y.to(device)
        Z = projector(X)     # (B,D)
        Z = inorm(Z)
        P = model(Z)
        ys.append(Y.detach().cpu())
        ps.append(P.detach().cpu())
        ids.extend(list(ks))
    if len(ys) == 0:
        return {"RMSE": float("nan"), "R2": float("nan"), "MAE": float("nan"), "Pearson": float("nan")}, np.array([]), np.array([])
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    # R2
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    # Pearson
    if len(y) >= 2:
        pear = float(np.corrcoef(y, p)[0, 1])
    else:
        pear = float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pear}, y, p, ids


def scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str):
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()


# -----------------------------
# Runner for one embedding type
# -----------------------------
def run_one_embedding(name: str,
                      emb_A_path: str,
                      smi_A_path: str,
                      emb_O_path: str,
                      smi_O_path: str,
                      split_path: str,
                      emb_type: str,
                      method: str) -> Dict:
    print(f"\n=== {name} ===")
    set_all_seeds(SEED)

    # Load embeddings
    emb_A = load_npz_dict(emb_A_path, key="emb")  # {id: np.array}
    emb_O = load_npz_dict(emb_O_path, key="emb")

    # Split
    with open(split_path, "r") as f:
        sp = json.load(f)
    tr_ids = list(sp.get("train", []))
    va_ids = list(sp.get('valid', sp.get('val', [])))
    te_ids = list(sp.get("test", []))

    # Intersect with embedding keys
    A_ids = set(emb_A.keys())
    O_ids = set(emb_O.keys())
    tr_ids = [k for k in tr_ids if k in A_ids]
    va_ids = [k for k in va_ids if k in A_ids]
    te_ids = [k for k in te_ids if k in O_ids]

    # Labels
    lab_A = parse_labels_by_ids(smi_A_path, set(tr_ids) | set(va_ids))
    lab_O = parse_labels_by_ids(smi_O_path, set(te_ids))

    # Filter to those with labels
    tr_ids = [k for k in tr_ids if k in lab_A]
    va_ids = [k for k in va_ids if k in lab_A]
    te_ids = [k for k in te_ids if k in lab_O]

    # Infer in_dim from one sample
    sample_A = emb_A[tr_ids[0]] if tr_ids else next(iter(emb_A.values()))
    if sample_A.ndim == 1:
        sample_A_2d = deflatten_to_TD(sample_A)
        in_dim = int(sample_A_2d.shape[1])
    else:
        in_dim = int(sample_A.shape[-1])

    # Projector
    projector = build_projector(in_dim=in_dim, emb_type=emb_type, method=method)

    # DataLoaders
    tr_set = PairDataset(emb_A, lab_A, tr_ids)
    va_set = PairDataset(emb_A, lab_A, va_ids)
    te_set = PairDataset(emb_O, lab_O, te_ids)

    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    te_loader = DataLoader(te_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector = projector.to(device)

    # Fit projector (e.g., ECFP PCA/UMAP/AE)
    try:
        projector.fit(tr_loader, device)
    except Exception as e:
        print("[WARN] projector.fit skipped or failed:", e)

    # Fit InputNorm on TRAIN features only
    inorm = InputNorm(PROJ_OUT_DIM).to(device)
    inorm.fit(tr_loader, projector, device)

    # Train head
    model, best_dict, curve_png = train_one(name, tr_loader, va_loader, projector, inorm, device)

    # Load best
    best_ckpt = torch.load(os.path.join(OUT_DIR, f"(2){name}_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])  # type: ignore

    # Eval
    metrics_va, y_true_va, y_pred_va, ids_va = eval_metrics(model, projector, inorm, va_loader, device)
    metrics_te, y_true_te, y_pred_te, ids_te = eval_metrics(model, projector, inorm, te_loader, device)

    # Save preds & plots
    import pandas as pd
    preds_csv_va = os.path.join(OUT_DIR, f"(2)preds_v2_{name}_VAL.csv")
    preds_csv_te = os.path.join(OUT_DIR, f"(2)preds_v2_{name}.csv")

    if y_true_va.size > 0:
        save_ranked_preds_csv(preds_csv_va, ids_va, y_true_va, y_pred_va)
    else:
        print("[WARN] VAL set empty → skip VAL csv/plot")
    
    save_ranked_preds_csv(preds_csv_te, ids_te, y_true_te, y_pred_te)
              
    _print_one_line_metrics("TEST", metrics_te, len(y_true_te))

    #pd.DataFrame({"y_true": y_true_va, "y_pred": y_pred_va}).to_csv(preds_csv_va, index=False)
    #pd.DataFrame({"y_true": y_true_te, "y_pred": y_pred_te}).to_csv(preds_csv_te, index=False)

    scatter_va = os.path.join(OUT_DIR, f"(2)pred_vs_true_v2_{name}_VAL.png")
    scatter_te = os.path.join(OUT_DIR, f"(2)pred_vs_true_v2_{name}.png")
    scatter_plot(y_true_va, y_pred_va, scatter_va, f"Pred vs True (VAL) — {name}")
    scatter_plot(y_true_te, y_pred_te, scatter_te, f"Pred vs True (TEST) — {name}")

    return {
        "val": {"best_val_loss": best_dict["best_val_loss"], **metrics_va},
        "test": metrics_te,
        "curve": curve_png,
        "preds_val": preds_csv_va,
        "preds_test": preds_csv_te,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    ensure_dir(OUT_DIR)
    set_all_seeds(SEED)

    results: Dict[str, Dict] = {}

    for name in TO_RUN:
        cfg = CFG[name]
        res = run_one_embedding(
            name=name,
            emb_A_path=cfg["emb_A"],
            smi_A_path=cfg["smi_A"],
            emb_O_path=cfg["emb_O"],
            smi_O_path=cfg["smi_O"],
            split_path=cfg["split"],
            emb_type=cfg["emb_type"],
            method=cfg["method"],
        )
        results[name] = res

    # Summary CSV
    import pandas as pd
    rows = []
    for k, v in results.items():
        rows.append({
            "name": k,
            "val_best_mse": v["val"]["best_val_loss"],
            "val_RMSE": v["val"].get("RMSE", float("nan")),
            "val_R2": v["val"].get("R2", float("nan")),
            "val_Pearson": v["val"].get("Pearson", float("nan")),
            "val_MAE": v["val"].get("MAE", float("nan")),
            "test_RMSE": v["test"]["RMSE"],
            "test_R2": v["test"]["R2"],
            "test_Pearson": v["test"]["Pearson"],
            "test_MAE": v["test"]["MAE"],
            "loss_curve_png": v["curve"],
            "preds_val_csv": v["preds_val"],
            "preds_test_csv": v["preds_test"],
        })
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_DIR, "(2)comparison_summary_v2.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Overlay scatter (TEST: ECFP vs GLEM)
    try:
        import pandas as pd
        def _load_preds(nm):
            df = pd.read_csv(os.path.join(OUT_DIR, f"(2)preds_v2_{nm}.csv"))
            return df["y_true"].values, df["y_pred"].values
        if all(os.path.exists(os.path.join(OUT_DIR, f"(2)preds_v2_{nm}.csv")) for nm in ["ECFP", "GLEM"]):
            yt_e, yp_e = _load_preds("ECFP")
            yt_g, yp_g = _load_preds("GLEM")
            mn = float(min(np.min(yt_e), np.min(yp_e), np.min(yt_g), np.min(yp_g)))
            mx = float(max(np.max(yt_e), np.max(yp_e), np.max(yt_g), np.max(yp_g)))
            plt.figure()
            plt.scatter(yt_e, yp_e, s=10, alpha=0.5, label="ECFP")
            plt.scatter(yt_g, yp_g, s=10, alpha=0.5, label="GLEM")
            plt.plot([mn, mx], [mn, mx])
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title("ECFP vs GLEM — Test Predictions")
            plt.legend()
            out_cmp = os.path.join(OUT_DIR, "(2)pred_vs_true_OVERLAY_v2_ECFP_vs_GLEM.png")
            plt.tight_layout()
            plt.savefig(out_cmp, dpi=400)
            plt.close()
            print(f"Saved: {out_cmp}")
    except Exception as e:
        print("overlay plot skipped:", e)


if __name__ == "__main__":
    main()
