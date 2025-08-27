import torch
import torch.nn as nn
from typing import Optional

# -----------------------------
# Shared blocks
# -----------------------------

class AttnPool(nn.Module):
    """Learned attention pooling. Keeps feature dim (D) unchanged.
    x: (B, T, D) or (B, D) -> returns (B, D)
    mask: (B, T) boolean optional
    """
    def __init__(self, d:int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim == 2:
            return x
        scores = (x * self.q).sum(dim=-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1)
        return (x*w.unsqueeze(-1)).sum(dim=1)
    
class _ProjPoolBase(nn.Module):
    """Project GNN/GLEM to same dim, then pool"""
    def __init__(self, d_gnn:int, d_glem: int, d_hidden:int=128):
        super().__init__()
        self.d_hidden = d_hidden
        self.proj_gnn = nn.Sequential(
            nn.Linear(d_gnn, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.proj_glem = nn.Sequential(
            nn.Linear(d_glem, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.pool_g = AttnPool(d_hidden)
        self.pool_t = AttnPool(d_hidden)

    @staticmethod
    def _ensure_bt_d(x: torch.Tensor) -> torch.Tensor:
        # Accept (B, D) or (B, T, D). If 2D, add T=1.
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x
        
    def _project_and_pool(self, G:torch.Tensor, T:torch.Tensor,
                          mask_gnn: Optional[torch.Tensor]=None,
                          mask_glem: Optional[torch.Tensor]=None):
        G = self._ensure_bt_d(G)
        T = self._ensure_bt_d(T)
        g_tok = self.proj_gnn(G)
        t_tok = self.proj_glem(T)
        g_pool = self.pool_g(g_tok, mask_gnn)
        t_pool = self.pool_t(t_tok, mask_glem)
        return g_pool, t_pool

# -----------------------------
# 1) EqualMix fusion (no gate)
# -----------------------------
class FusionEqualMix(_ProjPoolBase):
    def __init__(self, d_gnn: int, d_glem: int, d_hidden: int = 128):
        super().__init__(d_gnn, d_glem, d_hidden)
        self.proj_final = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )

    def forward(self, G:torch.Tensor, T:torch.Tensor,
                mask_gnn: Optional[torch.Tensor]=None,
                mask_glem: Optional[torch.Tensor]=None):
        g_pool, t_pool  = self._project_and_pool(G, T, mask_gnn, mask_glem)
        fused_avg = 0.5 * g_pool + 0.5 * t_pool
        fused = self.proj_final(torch.cat([fused_avg, g_pool, t_pool], dim=-1))
        gate = torch.full_like(g_pool, 0.5)
        return fused, gate

# -----------------------------
# 2) Concat fusion (no gate)
# -----------------------------
class FusionConcat(_ProjPoolBase):
    def __init__(self, d_gnn: int, d_glem: int, d_hidden: int = 128):
        super().__init__(d_gnn, d_glem, d_hidden)
        # concat (g_pool, t_pool) -> (B, 2*Dh)
        self.proj_final = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )
    
    def forward(self, G:torch.Tensor, T:torch.Tensor,
                    mask_gnn: Optional[torch.Tensor]=None,
                    mask_glem: Optional[torch.Tensor]=None):
        g_pool, t_pool = self._project_and_pool(G, T, mask_gnn, mask_glem)
        fused = self.proj_final(torch.cat([g_pool, t_pool], dim=-1))
        gate = torch.full_like(g_pool, 0.5)
        return fused, gate
    
# -----------------------------
# 3) Gated fusion (scalar gate with smoothing)
# -----------------------------
class FusionGated(_ProjPoolBase):
    def __init__(self, d_gnn: int, d_glem: int, d_hidden: int = 128,
                 gate_clamp = (0.3, 0.7)):
        super().__init__(d_gnn, d_glem, d_hidden)
        self.gate_net = nn.Sequential(
            nn.Linear(2*d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1)
        )
        self.proj_final = nn.Sequential(
            nn.Linear(3*d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )
        self.gate_lo, self.gate_hi = gate_clamp

    def forward(self, G:torch.Tensor, T:torch.Tensor,
                    mask_gnn: Optional[torch.Tensor]=None,
                    mask_glem: Optional[torch.Tensor]=None):
        g_pool, t_pool = self._project_and_pool(G, T, mask_gnn, mask_glem)
        gate_s = torch.sigmoid(self.gate_net(torch.cat([g_pool, t_pool], dim=-1))).clamp(self.gate_lo, self.gate_hi)
        gate = gate_s.expand_as(g_pool)
        fused_avg = gate * g_pool + (1.0 -gate) * t_pool
        fused = self.proj_final(torch.cat([fused_avg, g_pool, t_pool], dim=-1))
        return fused, gate
    
# -----------------------------
# Factory
# -----------------------------
FUSION_MAP = {
    "equalmix": FusionEqualMix,
    "concat": FusionConcat,
    "gated": FusionGated
}

def make_fusion(kind:str, d_gnn:int, d_glem:int, d_hidden:int=128, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind not in FUSION_MAP:
        raise ValueError(f"Unknown fusion kind: {kind}. Choose from {list(FUSION_MAP.keys())}")
    return FUSION_MAP[kind](d_gnn=d_gnn, d_glem=d_glem, d_hidden=d_hidden, **kwargs)
