
"""
Build proper ECFP4 (Morgan radius=2, nBits=2048) from an SMI file and save to NPZ.
SMI format: <SMILES><ws><ID><ws><Solubility> (headerless; extra columns ignored).

Usage:
  python build_ecfp_from_smi.py A_glem.smi --n_bits 2048 --radius 2 -o A_ecfp4_fresh.npz

Requires RDKit (e.g., `pip install rdkit-pypi` or conda rdkit).
"""
import argparse, sys, re
import numpy as np

def read_smi(path):
    smiles, ids = [], []
    # utf-8-sig handles BOM if present
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            if len(p) < 2:
                continue
            smiles.append(p[0])
            ids.append(p[1])
    return smiles, ids

def build_ecfp(smiles, n_bits=2048, radius=2, use_features=False):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as e:
        print("RDKit is required. Install rdkit-pypi or conda rdkit.", file=sys.stderr)
        raise

    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    valid = np.ones(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            valid[i] = False
            continue
        if use_features:
            fp = Chem.MorganGenerator(m, radius, nBits=n_bits, useFeatures=True)
        else:
            fp = Chem.MorganGenerator(m, radius, nBits=n_bits)
        on = list(fp.GetOnBits())
        if len(on) == 0:
            # leave as all-zeros (rare), filter later if needed
            pass
        X[i, on] = 1
    return X, valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smi", help="input .smi (SMILES ID [label])")
    ap.add_argument("-o","--output", default=None, help="output npz path (default: <stem>_ecfp{2*radius}.npz)")
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--use_features", action="store_true", help="use feature invariants (ECFP-like)")
    args = ap.parse_args()

    smiles, ids = read_smi(args.smi)
    if not smiles:
        print("No rows read from SMI. Check file/encoding.", file=sys.stderr); sys.exit(2)

    X, valid = build_ecfp(smiles, n_bits=args.n_bits, radius=args.radius, use_features=args.use_features)
    smiles = [s for s,v in zip(smiles, valid) if v]
    ids = [i for i,v in zip(ids, valid) if v]
    X = X[valid]

    out = args.output or re.sub(r"\.smi$", "", args.smi) + f"_ecfp{args.radius*2}.npz"
    np.savez_compressed(out, emb=X.astype(np.float32), ids=np.array(ids, dtype=object))
    # quick sanity stats
    nnz_row = (X!=0).sum(1)
    print(f"[done] {out}  shape={X.shape}  row nnz min/median/mean/max="
          f"{int(nnz_row.min())}/{float(np.median(nnz_row)):.1f}/{float(nnz_row.mean()):.1f}/{int(nnz_row.max())}")
if __name__ == "__main__":
    main()
