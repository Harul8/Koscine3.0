"""
Denoising Autoencoder + MLP model (PyTorch).

Architecture motivation:
  Financial features have high noise-to-signal ratio.  The AE bottleneck
  compresses the input to a low-dimensional latent space, implicitly filtering
  noise by forcing the encoder to retain only predictive structure.  The MLP
  head then concatenates the original (cleaned) input with the latent
  representation to predict the binary target.

Architecture (per forward pass during training):
  1. RobustScaler — fit on train, applied to all splits
  2. NaN → 0 imputation
  3. Gaussian noise injection  (train only)
  4. BatchNorm + Encoder: n → enc_dim → latent_dim
  5. Decoder: latent_dim → enc_dim → n  (reconstruction)
  6. AE supervised head: latent_dim → 1  (BCE)
  7. MLP head: concat(x_clean, latent) → mlp_dim → mlp_dim//2 → 1  (BCE)

Three simultaneous losses:
  L = MSE(reconstruction, x_clean) / n_features   [scale-invariant reconstruction]
    + BCE(ae_logit, y)
    + BCE(mlp_logit, y)

Early stopping on val Average Precision (MLP head only).
"""
from __future__ import annotations
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


@contextmanager
def _nullctx():
    yield

from .base import BaseModel


# ── PyTorch module ─────────────────────────────────────────────────────────────

def _build_net(n_features: int, enc_dim: int, latent_dim: int,
               mlp_dim: int, dropout: float):
    """Build and return the AE+MLP module."""
    import torch
    import torch.nn as nn

    class AEMLP(nn.Module):
        def __init__(self):
            super().__init__()
            act = nn.SiLU

            # Input normalisation
            self.bn_in = nn.BatchNorm1d(n_features)

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(n_features, enc_dim), act(),
                nn.Linear(enc_dim, latent_dim),  act(),
            )

            # Decoder (reconstruction)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, enc_dim), act(),
                nn.Linear(enc_dim, n_features),
            )

            # AE supervised head (predict target from latent alone)
            self.ae_head = nn.Linear(latent_dim, 1)

            # MLP head (concat of original + latent)
            self.mlp = nn.Sequential(
                nn.Linear(n_features + latent_dim, mlp_dim), act(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, mlp_dim // 2), act(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim // 2, 1),
            )

        def forward(self, x, noise_std: float = 0.0):
            x_norm  = self.bn_in(x)
            x_noisy = x_norm + torch.randn_like(x_norm) * noise_std
            z       = self.encoder(x_noisy)
            x_hat   = self.decoder(z)
            ae_logit  = self.ae_head(z).squeeze(-1)
            mlp_logit = self.mlp(torch.cat([x_norm, z], dim=-1)).squeeze(-1)
            return x_hat, x_norm, ae_logit, mlp_logit

    return AEMLP()


# ── Model class ────────────────────────────────────────────────────────────────

class AEMLPModel(BaseModel):
    model_type = "ae_mlp"

    def __init__(
        self,
        target: str,
        feature_cols: list[str] | None = None,
        enc_dim:    int   = 256,
        latent_dim: int   = 32,
        mlp_dim:    int   = 256,
        dropout:    float = 0.20,
        noise_std:  float = 0.10,
        lr:         float = 1e-3,
        n_epochs:   int   = 200,
        batch_size: int   = 16384,
        patience:   int   = 50,
    ):
        self.target                 = target
        self._feature_cols_override = feature_cols
        self.enc_dim    = enc_dim
        self.latent_dim = latent_dim
        self.mlp_dim    = mlp_dim
        self.dropout    = dropout
        self.noise_std  = noise_std
        self.lr         = lr
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.patience   = patience

        self._scaler    = None
        self._net       = None
        self._device    = None
        self._feat_cols: list[str] = []
        self._base_rate: float = float("nan")

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import RobustScaler

        self._feat_cols = self._numeric_feat_cols(X_train, self._feature_cols_override)
        self._base_rate = float(np.asarray(y_train).mean())

        # Preprocessing: impute NaN → 0, scale
        Xtr = self._clean(X_train[self._feat_cols]).fillna(0.0).values.astype(np.float32)
        Xva = self._clean(X_val[self._feat_cols]).fillna(0.0).values.astype(np.float32)
        ytr = np.asarray(y_train, dtype=np.float32)
        yva = np.asarray(y_val,   dtype=np.float32)

        self._scaler = RobustScaler()
        Xtr = self._scaler.fit_transform(Xtr).astype(np.float32)
        Xva = self._scaler.transform(Xva).astype(np.float32)

        n_feat = Xtr.shape[1]
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [ae_mlp:{self.target}] device={self._device}  "
              f"n_feat={n_feat}  train={len(Xtr):,}  val={len(Xva):,}")

        self._net = _build_net(n_feat, self.enc_dim, self.latent_dim,
                               self.mlp_dim, self.dropout).to(self._device)

        # Class-imbalance weight for BCE
        pos_w = (1 - self._base_rate) / (self._base_rate + 1e-7)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self._device))
        mse = nn.MSELoss()
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_epochs)

        Xtr_t = torch.from_numpy(Xtr).to(self._device)
        ytr_t = torch.from_numpy(ytr).to(self._device)
        Xva_t = torch.from_numpy(Xva).to(self._device)
        # yva stays on CPU — sklearn AP/AUC expect numpy

        use_amp = (self._device == "cuda")
        scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Per-thread CUDA stream so parallel Optuna trials don't share the
        # default stream and serialize — each trial's kernels can overlap.
        stream = torch.cuda.Stream() if self._device == "cuda" else None

        best_ap, best_state, no_improve = -1.0, None, 0

        for epoch in range(self.n_epochs):
            self._net.train()
            # CPU randperm avoids CUDA RNG race condition with n_jobs=4 threads
            idx = torch.randperm(len(Xtr_t)).to(self._device)
            for i in range(0, len(idx), self.batch_size):
                b_idx = idx[i: i + self.batch_size]
                xb, yb = Xtr_t[b_idx], ytr_t[b_idx]
                ctx = torch.cuda.stream(stream) if stream else _nullctx()
                with ctx, torch.autocast(device_type="cuda", enabled=use_amp):
                    x_hat, x_norm, ae_logit, mlp_logit = self._net(xb, self.noise_std)
                    loss = (mse(x_hat, x_norm) / n_feat
                            + bce(ae_logit, yb)
                            + bce(mlp_logit, yb))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sch.step()

            # Validate every 10 epochs — halves AP overhead vs every-5
            if (epoch + 1) % 10 == 0:
                self._net.eval()
                ctx = torch.cuda.stream(stream) if stream else _nullctx()
                with ctx, torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                    _, _, _, vl = self._net(Xva_t, 0.0)
                    vprob = torch.sigmoid(vl).cpu().float().numpy()
                from sklearn.metrics import average_precision_score
                ap = average_precision_score(yva, vprob)
                if ap > best_ap:
                    best_ap = ap
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 10
                if no_improve >= self.patience:
                    print(f"    early stop at epoch {epoch+1}  best_ap={best_ap:.4f}")
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        # Final val metrics
        self._net.eval()
        with torch.no_grad():
            _, _, _, vl = self._net(Xva_t, 0.0)
            val_probs = torch.sigmoid(vl).cpu().float().numpy()

        metrics = self._metrics(yva, val_probs)
        print(f"  [{self.target}] val_auc={metrics['val_auc']:.3f}  "
              f"val_ap={metrics['val_ap']:.3f}  "
              f"prec@10%={metrics['val_prec10']:.2%}")
        return metrics

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame, infer_batch: int = 8192) -> np.ndarray:
        import torch
        Xin = self._clean(X[self._feat_cols]).fillna(0.0).values.astype(np.float32)
        Xin = self._scaler.transform(Xin).astype(np.float32)
        dev = self._device or "cpu"
        self._net.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(Xin), infer_batch):
                Xt = torch.from_numpy(Xin[i: i + infer_batch]).to(dev)
                _, _, _, logit = self._net(Xt, 0.0)
                chunks.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(chunks)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict":            self._net.state_dict(),
            "scaler":                self._scaler,
            "feat_cols":             self._feat_cols,
            "base_rate":             self._base_rate,
            "target":                self.target,
            "model_type":            self.model_type,
            "feature_cols_override": self._feature_cols_override,
            "hparams": dict(
                enc_dim=self.enc_dim, latent_dim=self.latent_dim,
                mlp_dim=self.mlp_dim, dropout=self.dropout,
            ),
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "AEMLPModel":
        import torch
        with open(path, "rb") as fh:
            p = pickle.load(fh)
        hp = p.get("hparams", {})
        m = cls(
            target=p["target"],
            enc_dim    = hp.get("enc_dim", 256),
            latent_dim = hp.get("latent_dim", 32),
            mlp_dim    = hp.get("mlp_dim", 256),
            dropout    = hp.get("dropout", 0.20),
        )
        m._feat_cols              = p["feat_cols"]
        m._base_rate              = p["base_rate"]
        m._scaler                 = p["scaler"]
        m._feature_cols_override  = p.get("feature_cols_override")
        m._device    = "cuda" if torch.cuda.is_available() else "cpu"

        n_feat = len(m._feat_cols)
        m._net = _build_net(n_feat, m.enc_dim, m.latent_dim,
                            m.mlp_dim, m.dropout).to(m._device)
        m._net.load_state_dict(p["state_dict"])
        m._net.eval()
        return m

    @property
    def base_rate(self) -> float:
        return self._base_rate

    @property
    def feat_cols(self) -> list[str]:
        return self._feat_cols
