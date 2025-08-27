import os
import json
import math
from typing import List, Dict, Tuple
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Helvetica Neue'

from FusionVariants import FusionGated  # attn-pool + scalar gate + clamp

# -----------------------------
# Config (edit paths as needed)
# -----------------------------
A_GNN = "./Embedding/A_egnn.emb.npz"
A_GLEM = "./Embedding/A_glem.emb.npz"
O_GNN = "./Embedding/O_egnn.emb.npz"
O_GLEM = "./Embedding/O_glem.emb.npz"
A_SMI = "./Embedding/A_glem.smi"  # labels for A split
O_SMI = "./Embedding/O_glem.smi"  # labels for O split
SPLIT_JSON = "./DataSplit/split.v2.json"

BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-4
WEIGHT_DECAY = 1e-2
D_H = 128
MID_DIM = 128
DROPOUT = 0.3
SEED = 42
PATIENCE = 20

# regularizers (very light)
LAMBDA_GATE = 1e-3   # penalize (a-0.5)^2 to prevent collapse
LAMBDA_ALIGN = 1e-3  # optional alignment of pooled features
GATE_CLAMP = (0.3, 0.7)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Utilities
# -----------------------------
np.random.seed(SEED); torch.manual_seed(SEED)

def load_npz_dict(path: str, key: str = "emb") -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)[key].item()
    return d

BASES = (128, 256)

def deflatten_to_TD(x: np.ndarray) -> np.ndarray:
    a = np.array(x)
    if a.ndim == 1:
        n = a.size
        for b in BASES:
            if n > b and n % b == 0:
                return a.reshape(n // b, b)
        return a.reshape(1, -1)  # (1,D) standalone vector
    return a  # (T,D) already


def parse_labels_by_ids(path: str, id_candidates: set) -> Dict[str, float]:
    """
    .smi(혹은 유사파일) 한 줄에서 'float로 해석되는 마지막 토큰'을 label로,
    그 줄의 토큰 중에서 NPZ 키 집합(id_candidates)에 '정확히 존재하는' 토큰을 ID로 채택.
    매칭되는 토큰이 없으면 해당 라인은 건너뜀.
    """
    id_set = {str(x) for x in id_candidates}
    lab: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 공백/탭/콤마 기준으로 나눔
            parts = [p for p in line.replace(",", " \t ").split() if p]
            if len(parts) < 2:
                continue
            # label = 마지막 float 토큰
            y = None
            for tok in reversed(parts):
                try:
                    y = float(tok)
                    break
                except Exception:
                    continue
            if y is None:
                continue
            # 이 줄에서 NPZ 키와 정확히 일치하는 토큰을 ID로 선택
            cand_ids = [tok for tok in parts if tok in id_set]
            if not cand_ids:
                continue
            _id = cand_ids[0]      # 일치 토큰이 여러 개면 첫 번째 사용
            lab[_id] = y
    return lab



class PairDataset(Dataset):
    def __init__(self, ids: List[str], gnn: Dict[str, np.ndarray], glem: Dict[str, np.ndarray], labels: Dict[str, float]):
        self.ids = ids
        self.gnn = gnn
        self.glem = glem
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        k = self.ids[idx]
        G = deflatten_to_TD(self.gnn[k])   # (T,D) or (1,D)
        T = deflatten_to_TD(self.glem[k])
        y = float(self.labels[k])
        return G.astype(np.float32), T.astype(np.float32), y, k


def pad_collate(batch):
    # batch: list of (G_np (Tg,D), T_np (Tt,D), y, k)
    Gs, Ts, ys, ks = zip(*batch)
    Tg = [g.shape[0] for g in Gs]
    Tt = [t.shape[0] for t in Ts]
    Dg = Gs[0].shape[-1]
    Dt = Ts[0].shape[-1]
    assert all(g.shape[-1]==Dg for g in Gs)
    assert all(t.shape[-1]==Dt for t in Ts)

    B = len(Gs)
    max_Tg = max(Tg)
    max_Tt = max(Tt)

    G = torch.zeros((B, max_Tg, Dg), dtype=torch.float32)
    T = torch.zeros((B, max_Tt, Dt), dtype=torch.float32)
    Mg = torch.zeros((B, max_Tg), dtype=torch.bool)
    Mt = torch.zeros((B, max_Tt), dtype=torch.bool)

    for i, (g, t) in enumerate(zip(Gs, Ts)):
        G[i, :g.shape[0], :] = torch.from_numpy(g)
        T[i, :t.shape[0], :] = torch.from_numpy(t)
        Mg[i, :g.shape[0]] = True
        Mt[i, :t.shape[0]] = True

    y = torch.tensor(ys, dtype=torch.float32)
    return G, T, Mg, Mt, y, ks


class InputNorm(nn.Module):
    """Feature-wise normalization (fit on TRAIN only) for fused features."""
    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))

    @torch.no_grad()
    def fit(self, loader: DataLoader, fusion: nn.Module, head: nn.Module):
        fusion.eval(); head.eval()
        n = 0; m = None; v = None
        for G, T, Mg, Mt, y, ks in loader:
            G = G.to(DEVICE); T = T.to(DEVICE)
            f, _ = fusion(G, T, Mg.to(DEVICE), Mt.to(DEVICE))  # (B,D)
            if m is None:
                m = f.sum(0)
                v = (f**2).sum(0)
            else:
                m += f.sum(0)
                v += (f**2).sum(0)
            n += f.shape[0]
        mu = m / max(1, n)
        var = v / max(1, n) - mu**2
        sigma = var.clamp_min(1e-6).sqrt()
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma


class Regressor(nn.Module):
    def __init__(self, d_in: int, mid: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, mid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid, mid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def eval_metrics(fusion, head, inorm, loader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    fusion.eval(); head.eval()
    ys, ps = [], []
    for G, T, Mg, Mt, y, ks in loader:
        G = G.to(DEVICE); T = T.to(DEVICE)
        f, gate = fusion(G, T, Mg.to(DEVICE), Mt.to(DEVICE))
        if inorm is not None:
            f = inorm(f)
        pred = head(f)
        ys.append(y.numpy())
        ps.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys).ravel()
    y_pred = np.concatenate(ps).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    # pearson (safe)
    yt = y_true - y_true.mean(); yp = y_pred - y_pred.mean()
    denom = float(np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()) + 1e-12)
    pr = float((yt*yp).sum() / denom) if denom > 0 else 0.0
    return {"RMSE": rmse, "R2": r2, "MAE": mae, "Pearson": pr}, y_true, y_pred

# --- Utilities to collect predictions (with IDs) and save CSV ---
@torch.no_grad()
def collect_predictions(fusion, head, inorm, loader):
    fusion.eval(); head.eval()
    ids, ys, ps = [], [], []
    for G, T, Mg, Mt, y, ks in loader:
        G = G.to(DEVICE); T = T.to(DEVICE)
        f, _ = fusion(G, T, Mg.to(DEVICE), Mt.to(DEVICE))
        if inorm is not None:
            f = inorm(f)
        pred = head(f)
        ys.append(y.numpy()); ps.append(pred.detach().cpu().numpy()); ids += list(ks)
    y_true = np.concatenate(ys).ravel()
    y_pred = np.concatenate(ps).ravel()
    return ids, y_true, y_pred

def save_predictions_csv(path_csv: str, ids, y_true, y_pred, topk_annot: int = 20):
    import os as _os
    _os.makedirs(_os.path.dirname(path_csv), exist_ok=True)
    resid = y_pred - y_true
    absr = np.abs(resid)
    order = absr.argsort()[::-1]  # descending by abs error
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","true","pred","residual","abs_residual","rank_by_abs_error"])
        for rank, idx in enumerate(order, start=1):
            w.writerow([ids[idx], float(y_true[idx]), float(y_pred[idx]), float(resid[idx]), float(absr[idx]), rank])
    return order[:min(topk_annot, len(order))]


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 1) load dicts & labels & split
    gA = load_npz_dict(A_GNN); eA = load_npz_dict(A_GLEM)
    gO = load_npz_dict(O_GNN); eO = load_npz_dict(O_GLEM)
    
    gA = load_npz_dict(A_GNN); eA = load_npz_dict(A_GLEM)
    gO = load_npz_dict(O_GNN); eO = load_npz_dict(O_GLEM)

    # NPZ 키 교집합(실제로 존재하는 샘플들)
    idsA = set(gA.keys()) & set(eA.keys())
    idsO = set(gO.keys()) & set(eO.keys())

    labA = parse_labels_by_ids(A_SMI, idsA)
    labO = parse_labels_by_ids(O_SMI, idsO)


    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)

    # intersect ids for each split
    tr_ids = [i for i in split["train"] if i in gA and i in eA and i in labA]
    va_ids = [i for i in split.get("valid", split.get("val", [])) if i in gA and i in eA and i in labA]
    te_ids = [i for i in split["test"] if i in gO and i in eO and i in labO]

    tr_set = PairDataset(tr_ids, gA, eA, labA)
    va_set = PairDataset(va_ids, gA, eA, labA)
    te_set = PairDataset(te_ids, gO, eO, labO)

    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    te_loader = DataLoader(te_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    print(f"sizes → train={len(tr_set)}, valid={len(va_set)}, test={len(te_set)}")

    # 2) model
    fusion = FusionGated(d_gnn=D_H, d_glem=D_H, d_hidden=D_H, gate_clamp=GATE_CLAMP).to(DEVICE)
    head = Regressor(d_in=D_H, mid=MID_DIM, dropout=DROPOUT).to(DEVICE)

    inorm = InputNorm(D_H).to(DEVICE)
    inorm.fit(tr_loader, fusion, head)

        # AdamW with proper weight decay exclusion
    no_decay_keys = ["bias", "LayerNorm.weight", "layer_norm", "bn", "norm"]
    decay, no_decay = [], []
    for n, p in list(fusion.named_parameters()) + list(head.named_parameters()):
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay_keys):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(param_groups, lr=LR)
    # LR on plateau scheduler (reduces LR by 10x if no val improvement for `SCHED_PATIENCE`)
    SCHED_PATIENCE = 3
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=SCHED_PATIENCE)
    mse = nn.MSELoss()

    # 3) train
    best_val = float('inf'); bad = 0
    train_losses, val_losses = [] , []

    for ep in range(1, EPOCHS+1):
        fusion.train(); head.train()
        running = 0.0; nobs = 0
        for G, T, Mg, Mt, y, ks in tr_loader:
            G = G.to(DEVICE); T = T.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            f, gate = fusion(G, T, Mg.to(DEVICE), Mt.to(DEVICE))   # f: (B,D)
            f = inorm(f)
            pred = head(f)
            loss = mse(pred, y)
            # regularizers
            a = gate.mean(dim=1, keepdim=True)   # approximate scalar
            loss = loss + LAMBDA_GATE * ((a - 0.55) ** 2).mean()
            # no alignment on tokens; align pooled features via fusion internals (optional off by default)
            # (We'd need g_pool and t_pool to add L2; skipped for simplicity)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fusion.parameters()) + list(head.parameters()), 1.0)
            opt.step()

            running += float(loss.item()) * y.numel()
            nobs += int(y.numel())
        tr_loss = running / max(1, nobs)

        # validation
        fusion.eval(); head.eval()
        running = 0.0; nobs = 0
        with torch.no_grad():
            for G, T, Mg, Mt, y, ks in va_loader:
                G = G.to(DEVICE); T = T.to(DEVICE); y = y.to(DEVICE)
                f, gate = fusion(G, T, Mg.to(DEVICE), Mt.to(DEVICE))
                f = inorm(f)
                pred = head(f)
                loss = mse(pred, y)
                running += float(loss.item()) * y.numel()
                nobs += int(y.numel())
        va_loss = running / max(1, nobs)
        # adjust LR if plateau
        sched.step(va_loss)

        train_losses.append(tr_loss); val_losses.append(va_loss)
        print(f"[Epoch {ep:03d}] train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss; bad = 0
            # save checkpoint
            torch.save(fusion.state_dict(), "./E2E_results_modified/(5)fusion_e2e_best.pt")
            torch.save(head.state_dict(),   "./E2E_results_modified/(5)head_e2e_best.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {ep}")
                break

    # Load best checkpoint before evaluation
    if os.path.exists("./E2E_results_modified/(6)fusion_e2e_best.pt"):
        fusion.load_state_dict(torch.load("./E2E_results_modified/(6)fusion_e2e_best.pt", map_location=DEVICE))
    if os.path.exists("./E2E_results_modified/(6)head_e2e_best.pt"):
        head.load_state_dict(torch.load("./E2E_results_modified/(6)head_e2e_best.pt", map_location=DEVICE))
    fusion.eval(); head.eval()
# 4) test metrics + plots
    # Collect predictions (with IDs) for VAL/TEST and save CSVs
    val_ids, y_true_val, y_pred_val = collect_predictions(fusion, head, inorm, va_loader)
    test_ids, y_true_test, y_pred_test = collect_predictions(fusion, head, inorm, te_loader)
    import os as _os
    _os.makedirs("./E2E_results_modified", exist_ok=True)
    _ = save_predictions_csv("./E2E_results_modified/(6)predictions_val.csv",  val_ids,  y_true_val,  y_pred_val,  topk_annot=0)
    topk_te = save_predictions_csv("./E2E_results_modified/(6)predictions_test.csv", test_ids, y_true_test, y_pred_test, topk_annot=20)
    # metrics (from arrays)
    test_metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_test, y_pred_test))),
        "R2":   float(r2_score(y_true_test, y_pred_test)),
        "MAE":  float(mean_absolute_error(y_true_test, y_pred_test)),
        "Pearson": float(((y_true_test - y_true_test.mean()) * (y_pred_test - y_pred_test.mean())).sum() /
                         (np.sqrt(((y_true_test - y_true_test.mean())**2).sum()) * np.sqrt(((y_pred_test - y_pred_test.mean())**2).sum()) + 1e-12))
    }
    print("[TEST] RMSE={RMSE:.6f} | R2={R2:.6f} | Pearson={Pearson:.6f} | MAE={MAE:.6f}".format(**test_metrics))

    # Loss curve plot
    epochs_axis = np.arange(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs_axis, train_losses, label="Train Loss", color = 'lightseagreen')
    plt.plot(epochs_axis, val_losses,   label="Val Loss", color = 'coral')
    plt.xlabel("Epoch", fontweight = 'bold'); plt.ylabel("Loss (MSE)", fontweight = 'bold')
    #plt.title("E2E Fusion \n AqSolDBc | Train/Val, OChemUnseen | Test — Loss", fontweight = 'bold')
    plt.legend()
    #annot = ("Test RMSE: {RMSE:.4f}\n" "Test R²:   {R2:.4f}\n" "Pearson r: {Pearson:.4f}").format(**test_metrics)
    #plt.gcf().text(0.70, 0.60, annot, bbox=dict(boxstyle="round", alpha=0.3))
    plt.tight_layout(); plt.savefig("./E2E_results_modified/(6)loss_curve_e2e.png", dpi=400); plt.close()

    # Scatter
    plt.figure(); plt.scatter(y_true_test, y_pred_test, s=12, alpha=0.7, facecolors = 'teal')
    minv = min(float(np.min(y_true_test)), float(np.min(y_pred_test)))
    maxv = max(float(np.max(y_true_test)), float(np.max(y_pred_test)))
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("True"); plt.ylabel("Predicted") #plt.title("E2E Fusion — Test Predictions vs. True", fontweight = 'bold')
    plt.tight_layout(); plt.savefig("./E2E_results_modified/(6)pred_vs_true_e2e.png", dpi=400); plt.close()

    print("Saved: loss_curve_e2e.png, pred_vs_true_e2e.png, fusion_e2e_best.pt, head_e2e_best.pt")
