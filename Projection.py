import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

class Identity(nn.Module):
    def forward(self, x):
        return x

class ECFPAutoEncoder(nn.Module):
    """
    고차원 희소 ECFP > 저차원 잠재공간으로 압축하는 간단 AutoEncoder.
    fit() 단계에서 비지도로 ECFP를 재구성하도록 학습시킨 뒤, 인코더 출력 z를 사용.
    """
    def __init__(self, in_dim: int, latent_dim: int =128, hidden: int =512, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, in_dim)
        )
    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return z, x_hat
    

class ECFPProjector(nn.Module):
    """
    ECFP 전처리 + 차원 축소 파이프라인:
        (선택) Binarize > (선택) L2 normalize > StandardScaler > {PCA | UMAP | AE} 
        > (sigma / mu) 정규화 > (선택) LayerNorm
    fit()은 train split에서만 호출하여 누수 방지. forward()는 배치 텐서 (B,D) 또는 (B, T, D) 입력을 처리.
    """
    def __init__(
            self,
            out_dim: int = 128,
            method: str = "pca",
            binarize: bool = True,
            l2norm: bool = True,
            add_layernorm: bool = True,
            ae_hidden: int = 512,
            ae_dropout: float = 0.1,
            ae_epochs: int = 5, 
            ae_lr: float = 1e-3,
            sequence_mode: bool = False
    ):
        super().__init__()
        method = method.lower()
        assert method in {"pca", "umap", "ae"}, "method must be 'pca', 'umap', or 'ae'."
        self.out_dim = out_dim
        self.method = method
        self.binarize = binarize
        self.l2norm = l2norm
        self.add_layernorm = add_layernorm
        self.sequence_mode = sequence_mode

        self.scaler = None
        self.pca = None
        self.umap_model = None
        self.ae = None

        self.register_buffer("fmean", torch.zeros(out_dim))
        self.register_buffer("fstd", torch.ones(out_dim))
        self.ln = nn.LayerNorm(out_dim) if add_layernorm else Identity()

        self.ae_hidden = ae_hidden
        self.ae_dropout = ae_dropout
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr

    @torch.no_grad()
    def _collect(self, loader, device, max_batches = None):
        xs, nb = [], 0
        for emb, mask, info in loader:
            x = emb.to(device)
            if x.dim() == 3:
                x = x.mean(dim=1)
            xs.append(x.float().cpu())
            nb += 1
            if max_batches is not None and nb >= max_batches:
                break
        return torch.cat(xs, dim=0) if xs else torch.empty(0)
    
    @torch.no_grad()
    def fit(self, loader, device):
        X = self._collect(loader, device)
        if X.numel() == 0:
            raise RuntimeError("No ECFP data collected to fit projector.")
        
        # 1) 희소/카운트 완화
        if self.binarize:
            X = (X>0).float()
        if self.l2norm:
            X = X /(X.norm(dim=-1, keepdim=True)+1e-6)
        
        # 2) 특성 표준화
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = torch.from_numpy(self.scaler.fit_transform(X.numpy())).float()

        # 3) 차원 축소
        if self.method == "pca":
            self.pca = PCA(n_components=self.out_dim, svd_solver="auto", whiten=False, random_state=42)
            Xz = torch.from_numpy(self.pca.fit_transform(Xs.numpy())).float()
        
        elif self.method == "umap":
            self.umap_model = umap.UMAP(
                n_components=self.out_dim, n_neighbors=15, min_dist=0.1,
                metric='cosine', random_state=42
            )
            Xz = torch.from_numpy(self.umap_model.fit_transform(Xs.numpy())).float()
        
        else:
            in_dim = Xs.shape[1]
            self.ae = ECFPAutoEncoder(in_dim, latent_dim=self.out_dim, hidden=self.ae_hidden, dropout=self.ae_dropout).to(device)
            opt = torch.optim.AdamW(self.ae.parameters(), lr=self.ae_lr, weight_decay=1e-4)
            bs, N = 512, Xs.shape[0]
            for _ in range(self.ae_epochs):
                perm = torch.randperm(N)
                for i in range(0, N, bs):
                    idx = perm[i:i+bs]
                    xb = Xs[idx].to(device)
                    z, xb_hat = self.ae(xb)
                    loss = F.mse_loss(xb_hat, xb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                    opt.step()
            with torch.no_grad():
                z, _ = self.ae(Xs.to(device))
                Xz = z.detach().cpu()
            
        # 4) 축소 특징 mu / sigma 정장(추론 시 정규화에 적용)
        mu = Xz.mean(dim=0)
        std = Xz.std(dim=0, unbiased=False).clamp_min(1e-3)
        self.fmean.copy_(mu)
        self.fstd.copy_(std)

    def _numpy_reduce(self, x_np):
        if self.scaler is None:
            x_np_std = x_np
        else:
            x_np_std = self.sclaer.transform(x_np)
       
        if self.method == "pca":
            x_np_red = self.pca.transform(x_np_std) if self.pca is not None else x_np_std
        elif self.method == "umap":
            x_np_red = self.umap_model.transform(x_np_std) if self.umap_model is not None else x_np_std
        else:
            if self.ae is None:
                x_np_red = x_np_std
            else:
                with torch.no_grad():
                    z, _ = self.ae(torch.from_numpy(x_np_std).float().to(next(self.ae.parameters()).device))
                    x_np_red = z.cpu().numpy()
        return x_np_red

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 x: (B, D) 또는 (B, T, D). 출력: (B, out_dim)
        - (필요시) mean-pooling
        - (선택) binnarize, L2 normlize
        - StandardScaler > {PCA|UMAP|AE}
        - 축소 특징 mu / sigma 정규화 (+ LayerNorm)
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x2 = x.reshape( B*T, D)
        else: x2 = x

        xb = x2.float()
        if self.binarize:
            xb = (xb > 0).float()
        if self.l2norm:
            xb = xb / (xb.norm(dim=-1, keepdim=True) + 1e-6)
        
        x_np = xb.detach().cpu().numpy()
        x_np = self._numpy_reduce(x_np)
        z = torch.from_numpy(x_np).to(x.device).float()

        z = (z - self.fmean) / self.fstd
        z = self.ln(z)

        if x.dim() == 3:
            B, T, _D = x.shape
            z = z.view(B, T, -1)
            if not self.sequence_mode:
                z = z.mean(dim=1)
            else:
                pass

        return z

class LinearProjector(nn.Module):
    def __init__(self, in_dim:int, out_dim: int=128, sequence_mode: bool=False):
        super().__init__()
        self.sequence_mode = sequence_mode
        self.proj = Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        if x.dim() == 3:
            if self.sequence_mode:
                x = self.proj(x.float())
            else:
                x = x.float().mean(dim=1)
                x = self.proj(x)
        else:
            x = self.proj(x.float())
        return x

class EmbeddingProjector(nn.Module):
    """
    Wrapper: emb_type에 'ecfp'가 들어가면 ECFPProjector 사용, 아니면 LinearProjector 사용
    """
    def __init__(self, in_dim: int, out_dim: int = 128, emb_type: str = "", method : str = "linear", sequence_mode: bool = False):
        super().__init__()
        self.emb_type = (emb_type or "").lower()
        self.out_dim = out_dim
        self.sequence_mode = sequence_mode

        if "ecfp" in self.emb_type:
            if method in {"pca", "umap", "ae"}:
                self.inner = ECFPProjector(out_dim=out_dim, method=method, sequence_mode=sequence_mode)
                self._mode = "ecfp"
            else:
                self.inner = LinearProjector(in_dim=in_dim, out_dim=out_dim, sequence_mode=sequence_mode)
                self._mode = "ecfp-linear"
        else:
            self.inner = LinearProjector(in_dim=in_dim, out_dim=out_dim, sequence_mode=sequence_mode)
            self._mode = "linear"
        
    # Projection.py 내부 예시
    @torch.no_grad()
    def fit(self, loader, device):
        feats = []
        for batch in loader:
            # 허용: (X, M, Y) 또는 (X, M, Y, ks)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    X = batch[0]
                else:
                    X = batch[0]  # 혹시 (X,) 형태면 그대로
            else:
                X = batch
            X = X.to(device)
            Z = self(X)             # projector forward
            feats.append(Z.detach().cpu())

        
    def forward(self, x):
        return self.inner(x)
