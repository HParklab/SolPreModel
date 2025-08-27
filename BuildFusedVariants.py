"""
Build three offline fused-embedding variants from A_/O_ EGNN+GLEM:
    1) equalmix  : f = 0.5*g + 0.5*e        -> {prefix}_fused_equalmix.emb.npz
    2) concat    : f = [g ; e] (axis=-1)    -> {prefix}_fused_concat.emb.npz
    3) gated(sc) : f = a*g + (1-a)*e, a=0.5 -> {prefix}_fused_gate.emb.npz
All outputs use key 'emb' and values shaepd (1, D*), float32.
This script is deterministic and does NOT use any random weights
"""
import numpy as np
import os
from typing import Dict

BASE_D = 128

def load_npz_dict(path:str, key:str="emb") -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)[key].item()
    return d

def deflatten(arr:np.ndarray, base: int=BASE_D) -> np.ndarray:
    a = np.array(arr)
    if a.ndim == 1 and a.size > base and a.size % base == 0:
        return a.reshape(-1, base)      # (T, D)
    if a.ndim == 1:
        return a.reshape(1, -1)         # (1, D)
    return a

def mean_pool(x: np.ndarray) -> np.ndarray:
    x = np.array(x)
    if x.ndim == 1:
        x = x[None, :]
    return x.mean(axis=0, keepdims=True)    # (1, D)

def build_one(prefix: str, out_dir: str="./Embedding_variants"):
    gnn = load_npz_dict(f"./Embedding/{prefix}_egnn.emb.npz", "emb")
    glem = load_npz_dict(f"./Embedding/{prefix}_glem.emb.npz", "emb")
    ids = sorted(set(gnn) & set(glem))

    eq, cat, gate = {}, {}, {}
    for k in ids:
        g = mean_pool(deflatten(gnn[k]))
        e = mean_pool(deflatten(glem[k]))
        eq[k] = (0.5*g + 0.5*e).astype(np.float32).copy()
        cat[k] = np.concatenate([g, e], axis=-1).astype(np.float32).copy()
        a = 0.5     # placeholder scalar gate
        gate[k] = (a*g + (1-a)*e).astype(np.float32).copy()

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_fused_equalmix.emb.npz"), emb=eq)
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_fused_concat.emb.npz"), emb=cat)
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_fused_gate.emb.npz"), emb=gate)
    print(f"[{prefix}] saved fused_equalmix / fused_concat / fused_gate in {out_dir}")

if __name__ == "__main__":
    for p in ("A", "O"):
        build_one(p, out_dir="./Embedding_variants")