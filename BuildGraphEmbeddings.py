
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem

import torch
import torch.nn as nn

from DataParser import DataParser
from egnn_norm import EGNN

# ---------- file utils ----------
def _nonempty_file(p: Optional[str]) -> bool:
    return (p is not None) and os.path.isfile(p) and os.path.getsize(p) > 0

# -----------------------------
# Helpers
# -----------------------------
def get_elements_from_smiles(smiles: str) -> List[str]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    return [a.GetSymbol() for a in m.GetAtoms()]


def bond_order_dict_from_smiles(smiles: str) -> Dict[Tuple[int, int], float]:
    # Return dict[(i,j)]=bond_order for both directions (0-based indices).
    d: Dict[Tuple[int, int], float] = {}
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return d
    Chem.Kekulize(m, clearAromaticFlags=False)
    for b in m.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        order = float(b.GetBondTypeAsDouble())  # aromatic -> 1.5
        d[(i, j)] = order
        d[(j, i)] = order
    return d


def fit_onehot_encoder(all_elements: List[str]) -> OneHotEncoder:
    # Two categorical features:
    #   col0: node_type in {'atom','grid'}
    #   col1: element_symbol in all_elements + ['GRID']
    elems_sorted = sorted(set(all_elements))
    categories = [
        np.array(['atom', 'grid'], dtype=object),
        np.array(elems_sorted + ['GRID'], dtype=object),
    ]
    enc = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    # Fit with a small design matrix that covers the categories
    demo = []
    for t in categories[0]:
        for e in categories[1]:
            demo.append([t, e])
    enc.fit(np.array(demo, dtype=object))
    return enc


def encode_nodes(enc: OneHotEncoder, node_types: List[str], element_symbols: List[str]) -> np.ndarray:
    X_cat = np.column_stack([np.array(node_types, dtype=object), np.array(element_symbols, dtype=object)])
    return enc.transform(X_cat)  # (N, K)


def knn_edges(coords_src: np.ndarray, coords_dst: np.ndarray, k: int, offset_src: int, offset_dst: int, same_set: bool=False) -> Tuple[List[int], List[int]]:
    # Torch-based KNN via topk on pairwise distances. If same_set=True, exclude self by inf-diagonal.
    if len(coords_src) == 0 or len(coords_dst) == 0 or k <= 0:
        return [], []
    src = torch.from_numpy(coords_src.astype(np.float32))
    dst = torch.from_numpy(coords_dst.astype(np.float32))
    k_eff = min(k, dst.shape[0])
    with torch.no_grad():
        # pairwise distances (n_src, n_dst)
        D = torch.cdist(src, dst, p=2)
        if same_set and D.shape[0] == D.shape[1]:
            D.fill_diagonal_(float('inf'))
        vals, idxs = torch.topk(D, k=k_eff, largest=False, dim=1)
        rows = torch.arange(src.shape[0], dtype=torch.long).repeat_interleave(k_eff) + offset_src
        cols = idxs.reshape(-1).long() + offset_dst
    return rows.tolist(), cols.tolist()


# -----------------------------
# Core pipeline
# -----------------------------
def build_graph_from_files(
    parser: DataParser,
    xyz_path: str,
    cosmo_path: str,
    smiles: str,
    enc: OneHotEncoder,
    k_atom_grid: int = 32,
    k_grid_grid: int = 8,
    include_global: bool = True,
    log_path: Optional[str] = None,
    wbo_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.LongTensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Build a heterogeneous graph (atoms + grids).
    # Node features: 
    # - atoms: [x,y,z] + partial_charge(from LOG) + OneHot(type, element)
    # - grids: [x,y,z] + electron_density(from COSMO) + onehot(type, "GRID")
    # Edges:
    #  - atom-atom: fully-connected (both directions), edge_attr=Wiberg bond order(from WBO) (nonbonded -> 0.0)
    #  - atom-grid: top-K nearest grids per atom (both directions), edge_attr = 0
    #  - grid-grid: top-K nearest neighbors per grid (both directions), edge_attr = 0
    # Returns: h0, x0, edge_index, edge_attr, gfeat (optional)

    # ------- Load geometry -------
    atom_coords = parser.load_xyz_coords(xyz_path)  # (Na, 3)
    # Expect: load_cosmo returns (grid_xyz, origin_atom_idx, electron_density)
    grid_coords, origin_idx, grid_density = parser.load_cosmo(cosmo_path)  # (Ng, 3), (Ng,), (Ng.)

    # ▶ 견고화: XYZ/GRID 표준화 + 빈 그리드 거르기
    atom_coords = np.asarray(atom_coords, dtype=np.float32)
    if atom_coords.ndim != 2 or atom_coords.shape[1] != 3 or atom_coords.shape[0] == 0:
        raise ValueError("XYZ parsed but has invalid shape or zero atoms")

    grid_coords = np.asarray(grid_coords, dtype=np.float32)
    if grid_coords.size == 0:
        # COSMO 파일은 있는데 그리드가 '아예 없음' → 이 분자는 skip
        raise ValueError("COSMO grid is empty")

    if grid_coords.ndim == 1:
        if grid_coords.shape[0] % 3 != 0:
            raise ValueError(f"COSMO grid malformed (flat len={grid_coords.shape[0]})")
        grid_coords = grid_coords.reshape(-1, 3)
    elif grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError(f"COSMO grid must be (Ng,3); got {grid_coords.shape}")


    Na = int(atom_coords.shape[0])
    Ng = int(grid_coords.shape[0])
    
    # Node one-hot (type + element)
    elements = get_elements_from_smiles(smiles)  # assume SMILES atom order aligns with XYZ atom order
    if len(elements) != Na:
        # If mismatch, we can still proceed by padding/truncating
        if len(elements) < Na:
            elements = elements + ['C'] * (Na - len(elements))
        else:
            elements = elements[:Na]
    node_types = ['atom'] * Na + ['grid'] * Ng
    node_elems = elements + ['GRID'] * Ng
    X_cat = encode_nodes(enc, node_types, node_elems).astype(np.float32)  # (Na+Ng, Kcat)

    # --- Numeric scalars per node: charge (atoms), electron density(grids) ---
    # LOG에서 atom partial charges
    if (log_path is not None) and os.path.isfile(log_path):
        try:
            #DataParser에 구현된 charge 파서 사용 (길이 Na의 np.ndarray 반환 가정)
            atom_charges = parser.load_xtb_charges(log_path)    # shape (Na,)
        except Exception:
            atom_charges = np.zeros(Na, dtype=np.float32)
    else:
        atom_charges = np.zeros(Na, dtype=np.float32)

    # COSMO에서 grid electron density
    if grid_density is None or len(grid_density) != Ng:
        grid_density = np.zeros(Ng, dtype=np.float32)
    atom_scalar = atom_charges.reshape(-1, 1).astype(np.float32)
    grid_scalar = np.asarray(grid_density, dtype=np.float32).reshape(-1, 1)


    # --- Coordinates ---
    x_atoms = atom_coords.astype(np.float32)
    x_grids = grid_coords.astype(np.float32)
    x0 = np.vstack([x_atoms, x_grids]).astype(np.float32)  # (N, 3)

    # --- Final node features: [x,y,z] + scalar(charge/density) + onehot ---
    node_scalar = np.vstack([atom_scalar, grid_scalar]).astype(np.float32)  # (N, 1)
    h0 = np.hstack([x0, node_scalar, X_cat]).astype(np.float32)  # (N, 4 + Kcat)

    # --- Edges & edge_attr ---
    rows: List[int] = []
    cols: List[int] = []
    edge_vals: List[float] = []

    # 1) atom-atom WBO (fallback: SMILES bond order)
    wbo_dict = None
    if (wbo_path is not None) and os.path.isfile(wbo_path):
        try:
            # DataParser.load_wbo(...) 는 dict[(i,j)]->float 를 반환
            wbo_dict = parser.load_wbo(wbo_path, make_symmetric=True)
        except Exception:
            wbo_dict = None

    bo_fallback = bond_order_dict_from_smiles(smiles)

    for i in range(Na):
        for j in range(Na):
            if i == j:
                continue
            if wbo_dict is not None:
                v = wbo_dict.get((i, j))
                if v is None:
                    v = bo_fallback.get((i, j), 0.0)
            else:
                v = bo_fallback.get((i, j), 0.0)
            rows.append(i); cols.append(j); edge_vals.append(float(v))

    
    bo_fallback = bond_order_dict_from_smiles(smiles)

    for i in range(Na):
        for j in range(Na):
            if i == j:
                continue
            if wbo_dict is not None:
                v = wbo_dict.get((i,j))
                if v is None:
                    v = bo_fallback.get((i,j), 0.0)
            else:
                v = bo_fallback.get((i,j), 0.0)
            rows.append(i); cols.append(j); edge_vals.append(float(v))

    # 2) atom-grid KNN (both directions), edge_attr = 0
    ag_r, ag_c = knn_edges(x_atoms, x_grids, k_atom_grid, offset_src=0, offset_dst=Na)
    rows += ag_r; cols += ag_c; edge_vals += [0.0] * len(ag_r)
    ga_r, ga_c = knn_edges(x_grids, x_atoms, k_atom_grid, offset_src=Na, offset_dst=0)
    rows += ga_r; cols += ga_c; edge_vals += [0.0] * len(ga_r)

    # 3) grid-grid KNN (both directions), edge_attr = 0
    gg_r, gg_c = knn_edges(x_grids, x_grids, k_grid_grid, offset_src=Na, offset_dst=Na)
    rows += gg_r; cols += gg_c; edge_vals += [0.0] * len(gg_r)

    # --- Tensors ---
    h0_t = torch.from_numpy(h0)
    x0_t = torch.from_numpy(x0)
    edge_index = [torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)]  # list of tensors
    
    edge_attr = None
    if len(edge_vals):
        edge_attr = torch.as_tensor(
            np.asarray(edge_vals, dtype=np.float32).reshape(-1,1)
        )
    # --- Global features (optional) ---
    gfeat = None
    if include_global and (log_path is not None) and os.path.isfile(log_path):
        try:
            quad, gsolv = parser.load_global_features(log_path)  # (6,), float
            quad = np.asarray(quad, dtype=np.float32).reshape(-1)
            gsolv = np.float32(gsolv)
            gfeat_np = np.concatenate([quad, np.array([gsolv], dtype=np.float32)], axis=0)
            gfeat = torch.from_numpy(gfeat_np)
        except Exception:
            gfeat = None

    return h0_t, x0_t, edge_index, edge_attr, gfeat


def run(
    csv_path: str,
    xyz_dir: str,
    cosmo_dir: str,
    out_npz: str,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    xyz_ext: str = ".xyz",
    cosmo_ext: str = ".cosmo",
    log_dir: Optional[str] = None,
    log_ext: str = ".log",
    wbo_dir: Optional[str] = None,
    wbo_ext: str = '_wbo',
    hidden_nf: int = 128,
    out_node_nf: int = 128,
    n_layers: int = 4,
    device: str = "cpu",
    include_global: bool = True,
    k_atom_grid: int = 32,
    k_grid_grid: int = 8,
):
    df = pd.read_csv(csv_path)
    assert smiles_col in df.columns and id_col in df.columns

    # Build OneHotEncoder from all elements in CSV
    all_elems: List[str] = []
    for smi in df[smiles_col].astype(str).tolist():
        all_elems.extend(get_elements_from_smiles(smi))
    enc = fit_onehot_encoder(all_elems)

    # EGNN: in_node_nf = 3 (xyz) + charge or density scalar + onehot_dim, in_edge_nf = 1 (bond order)
    onehot_dim = enc.transform(np.array([['atom', 'C']], dtype=object)).shape[1]
    in_node_nf = 4 + onehot_dim
    in_edge_nf = 1
    egnn = EGNN(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        out_node_nf=out_node_nf,
        in_edge_nf=in_edge_nf,
        device=device,
        n_layers=n_layers,
        node_agg='mean',
    )
    egnn.to(device)

    parser = DataParser(index_base='auto', return_base=0)

    emb_dict: Dict[str, np.ndarray] = {}
    fail_ids: List[str] = []

    for _, row in df.iterrows():
        mol_id = str(row[id_col])
        smi = str(row[smiles_col])

        xyz_path = os.path.join(xyz_dir, mol_id + xyz_ext)
        cosmo_path = os.path.join(cosmo_dir, mol_id + cosmo_ext)
        log_path = os.path.join(log_dir, mol_id + log_ext) if log_dir else None
        wbo_path = os.path.join(wbo_dir, mol_id + wbo_ext) if wbo_dir else None

        required_paths = [xyz_path, cosmo_path]
        if log_dir: required_paths.append(log_path)
        if wbo_dir: required_paths.append(wbo_path)

        if not all(_nonempty_file(p) for p in required_paths):
            fail_ids.append(mol_id)
            print(f"[WARN] Skipped {mol_id}: missing or empty required file(s)")
            continue

        try:
            h0, x0, edge_index, edge_attr, gfeat = build_graph_from_files(
                parser, xyz_path, cosmo_path, smi, enc,
                k_atom_grid=k_atom_grid, k_grid_grid=k_grid_grid,
                include_global=include_global, log_path=log_path, wbo_path = wbo_path
            )

            # Move to device & forward
            h0 = h0.to(device); x0 = x0.to(device)
            edge_index = [e.to(device) for e in edge_index]
            ea = edge_attr.to(device) if edge_attr is not None else None

            # Forward (EGNN forward expects: h, edge_index, coord, node_attr, edge_attr)
            with torch.no_grad():
                h_out, coord_out = egnn(h=h0, x=x0, edges=edge_index, edge_attr=ea)
                g_emb = h_out
                if include_global and (gfeat is not None):
                    g_all = torch.cat([g_emb.cpu(), gfeat], dim=0)
                else:
                    g_all = g_emb.cpu()

            emb_dict[mol_id] = g_all.numpy().astype(np.float32)

        except Exception as ex:
            fail_ids.append(mol_id)
            print(f"[WARN] Skipped {mol_id}: {ex}")

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, emb=emb_dict)

    meta = {
        "in_node_nf": in_node_nf,
        "in_edge_nf": in_edge_nf,
        "hidden_nf": hidden_nf,
        "out_node_nf": out_node_nf,
        "n_layers": n_layers,
        "device": device,
        "include_global": include_global,
        "k_atom_grid": k_atom_grid,
        "k_grid_grid": k_grid_grid,
        "num_success": len(emb_dict),
        "num_failed": len(fail_ids),
        "failed_ids": fail_ids,
    }
    with open(os.path.splitext(out_npz)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved embeddings for {len(emb_dict)} molecules to {out_npz}")
    if fail_ids:
        print(f"Failed {len(fail_ids)} IDs. See meta JSON for details.")


def main():
    ap = argparse.ArgumentParser(description="Build EGNN graph embeddings (atoms+grids) and save to NPZ.")
    ap.add_argument("--csv", required=True, help="CSV with at least ID and SMILES columns.")
    ap.add_argument("--xyz_dir", required=True, help="Directory containing {ID}.xyz files.")
    ap.add_argument("--cosmo_dir", required=True, help="Directory containing {ID}.cosmo files ($segment_information).")
    ap.add_argument("--out_npz", required=True, help="Output NPZ path to save {'emb': {ID: vec}}.")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--xyz_ext", default=".xyz")
    ap.add_argument("--cosmo_ext", default=".cosmo")
    ap.add_argument("--log_dir", default=None, help="Optional xTB log dir for global features (quadrupole+Gsolv).")
    ap.add_argument("--log_ext", default=".log")
    ap.add_argument("--wbo_dir", default=None, help="Directory containing {ID}_wbo files (Wiberg Bond Orders)")
    ap.add_argument("--wbo_ext", default="_wbo")
    ap.add_argument("--hidden_nf", type=int, default=128)
    ap.add_argument("--out_node_nf", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_global", action="store_true", help="Disable using global features even if logs exist.")
    ap.add_argument("--k_atom_grid", type=int, default=32)
    ap.add_argument("--k_grid_grid", type=int, default=8)

    args = ap.parse_args()

    run(
        csv_path=args.csv,
        xyz_dir=args.xyz_dir,
        cosmo_dir=args.cosmo_dir,
        out_npz=args.out_npz,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        xyz_ext=args.xyz_ext,
        cosmo_ext=args.cosmo_ext,
        log_dir=args.log_dir,
        log_ext=args.log_ext,
        wbo_dir=args.wbo_dir,
        wbo_ext=args.wbo_ext,
        hidden_nf=args.hidden_nf,
        out_node_nf=args.out_node_nf,
        n_layers=args.n_layers,
        device=args.device,
        include_global=(not args.no_global),
        k_atom_grid=args.k_atom_grid,
        k_grid_grid=args.k_grid_grid,
    )


if __name__ == "__main__":
    main()
