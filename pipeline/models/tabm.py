"""
TabM — parameter-efficient ensemble of MLPs (PyTorch).

Concept from "TabM: Advancing Tabular Deep Learning with Parameter-Efficient
Ensembling" (Gorishniy et al., 2024).

Implementation uses BatchEnsemble: k MLP members share a single weight matrix W,
each member i has per-element input scale R_i and output scale S_i, so the
effective weight for member i is diag(S_i) @ W @ diag(R_i).
This gives k independent members at the cost of one shared matrix + 2·k vectors.

Architecture per member:
  x → [shared embed] → BatchEnsemble layers → member head → logit_i
Final prediction: sigmoid(mean(logit_i))

Preprocessing: RobustScaler (fit on train).
Training: Adam + CosineAnnealing, early stopping on val AP.
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


# ── BatchEnsemble layer ────────────────────────────────────────────────────────

def _build_net(n_features: int, hidden_dim: int, n_layers: int,
               k: int, dropout: float):
    """Build and return a TabM network."""
    import torch
    import torch.nn as nn

    class BELinear(nn.Module):
        """Linear layer with BatchEnsemble: k members, one shared W."""
        def __init__(self, in_f: int, out_f: int, k: int):
            super().__init__()
            self.W    = nn.Linear(in_f, out_f, bias=False)
            self.R    = nn.Parameter(torch.ones(k, in_f))    # input scale
            self.S    = nn.Parameter(torch.ones(k, out_f))   # output scale
            self.bias = nn.Parameter(torch.zeros(k, out_f))
            # Init R and S with small random perturbations so members diverge
            nn.init.normal_(self.R, 1.0, 0.05)
            nn.init.normal_(self.S, 1.0, 0.05)

        def forward(self, x):
            # x: (k, B, in_f)
            k, B, _ = x.shape
            x_scaled = x * self.R.unsqueeze(1)              # (k, B, in_f)
            out = self.W(x_scaled.view(k * B, -1))           # (k*B, out_f)
            out = out.view(k, B, -1)
            out = out * self.S.unsqueeze(1) + self.bias.unsqueeze(1)
            return out                                        # (k, B, out_f)

    class TabM(nn.Module):
        def __init__(self):
            super().__init__()
            act = nn.SiLU

            # Shared input embedding (all members share this)
            self.embed = nn.Sequential(
                nn.BatchNorm1d(n_features),
                nn.Linear(n_features, hidden_dim),
                act(),
            )

            # BatchEnsemble hidden layers
            layers = []
            for _ in range(n_layers - 1):
                layers += [BELinear(hidden_dim, hidden_dim, k), act()]
            self.be_layers = nn.ModuleList(layers)
            self.dropout = nn.Dropout(dropout)

            # Per-member output head (no sharing)
            self.head = BELinear(hidden_dim, 1, k)

        def forward(self, x):
            # x: (B, n_features)
            B = x.shape[0]
            h = self.embed(x)                 # (B, hidden_dim)
            h = h.unsqueeze(0).expand(k, -1, -1)  # (k, B, hidden_dim)

            for layer in self.be_layers:
                if isinstance(layer, BELinear):
                    h = layer(h)
                else:
                    # activation or dropout — apply to reshaped tensor
                    kk, bbb, d = h.shape
                    h = layer(h.view(kk * bbb, d)).view(kk, bbb, d)

            h = self.dropout(h.view(k * B, -1)).view(k, B, -1)
            logits = self.head(h).squeeze(-1)   # (k, B)
            return logits.mean(dim=0)           # (B,) — average across members

    return TabM()


# ── Model class ────────────────────────────────────────────────────────────────

class TabMModel(BaseModel):
    model_type = "tabm"

    def __init__(
        self,
        target:       str,
        feature_cols: list[str] | None = None,
        k:          int   = 8,       # ensemble size
        hidden_dim: int   = 256,
        n_layers:   int   = 3,
        dropout:    float = 0.15,
        lr:         float = 1e-3,
        n_epochs:   int   = 200,
        batch_size: int   = 16384,
        patience:   int   = 50,
        seed:       int   = 42,      # for reproducibility + seed-ensemble training
    ):
        self.target                 = target
        self._feature_cols_override = feature_cols
        self.k          = k
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.lr         = lr
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.seed       = seed

        self._scaler    = None
        self._net       = None
        self._device    = None
        self._feat_cols: list[str] = []
        self._base_rate: float = float("nan")
        # Post-scaling clip: RobustScaler output is clipped to [-scaled_clip, +scaled_clip].
        # This neutralises catastrophic outliers from near-zero-IQR features (e.g.
        # pcr_chg_5d max|scaled|=192M) without touching informative extremes in
        # z-score/rank features (max|scaled| ≈ 5–15, well inside the clip bound).
        # The old pre-scaling winsorization at [p1,p99] destroyed signal in the
        # most overbought / most stretched stocks — exactly the dn_5_xl target cases.
        self._scaled_clip: float = 100.0

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import RobustScaler

        # Seed everything for reproducibility across seeds (different seed values
        # produce different but deterministic runs — enables seed-ensemble averaging).
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._feat_cols = self._numeric_feat_cols(X_train, self._feature_cols_override)
        self._base_rate = float(np.asarray(y_train).mean())

        Xtr = self._clean(X_train[self._feat_cols]).fillna(0.0).values.astype(np.float32)
        Xva = self._clean(X_val[self._feat_cols]).fillna(0.0).values.astype(np.float32)
        ytr = np.asarray(y_train, dtype=np.float32)
        yva = np.asarray(y_val,   dtype=np.float32)

        self._scaler = RobustScaler()
        Xtr = self._scaler.fit_transform(Xtr).astype(np.float32)
        Xva = self._scaler.transform(Xva).astype(np.float32)

        # ── Post-scaling clip ───────────────────────────────────────────────
        # Clamp after RobustScaler so near-zero-IQR features (scale_≈1, raw
        # outliers pass through as-is) cannot produce 100M+ scaled values.
        # Normal features (z-scores, ranks) have max|scaled|≈5–15 → unaffected.
        Xtr = np.clip(Xtr, -self._scaled_clip, self._scaled_clip)
        Xva = np.clip(Xva, -self._scaled_clip, self._scaled_clip)

        n_feat = Xtr.shape[1]
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [tabm:{self.target}] device={self._device}  k={self.k}  "
              f"hidden={self.hidden_dim}  train={len(Xtr):,}  val={len(Xva):,}")

        self._net = _build_net(n_feat, self.hidden_dim, self.n_layers,
                               self.k, self.dropout).to(self._device)

        pos_w = (1 - self._base_rate) / (self._base_rate + 1e-7)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self._device))
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        # Warm restarts: lr resets to max every T_0 epochs (×T_mult each cycle).
        # Allows escaping local minima that form early in training — the lr spike
        # at each restart kicks the model out of a shallow basin.
        # Cycle schedule with n_epochs=200: 0→50 (decay), 50→150 (decay), 150→200
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=50, T_mult=2, eta_min=self.lr * 0.01
        )

        Xtr_t = torch.from_numpy(Xtr).to(self._device)
        ytr_t = torch.from_numpy(ytr).to(self._device)
        Xva_t = torch.from_numpy(Xva).to(self._device)
        # yva stays on CPU — sklearn AP/AUC expect numpy
        yva_np = yva

        use_amp = (self._device == "cuda")
        scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

        # NOTE: previously used `torch.cuda.Stream()` per fit() for Optuna parallelism
        # but it deadlocks with GradScaler when fit() is called sequentially N times
        # (seed-ensemble training). For Optuna parallelism use n_jobs at process level.

        best_ap, best_state, no_improve = -1.0, None, 0

        for epoch in range(self.n_epochs):
            self._net.train()
            # CPU randperm avoids CUDA RNG race condition with n_jobs=4 threads
            idx = torch.randperm(len(Xtr_t)).to(self._device)
            for i in range(0, len(idx), self.batch_size):
                b = idx[i: i + self.batch_size]
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logit = self._net(Xtr_t[b])
                    loss  = bce(logit, ytr_t[b])
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sch.step()

            if (epoch + 1) % 10 == 0:
                self._net.eval()
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                    vprob = torch.sigmoid(self._net(Xva_t)).cpu().float().numpy()
                from sklearn.metrics import average_precision_score
                ap = average_precision_score(yva_np, vprob)
                if ap > best_ap:
                    best_ap = ap
                    best_state = {k2: v.clone() for k2, v in self._net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 10
                if no_improve >= self.patience:
                    print(f"    early stop at epoch {epoch+1}  best_ap={best_ap:.4f}")
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._net.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(self._net(Xva_t)).cpu().float().numpy()

        metrics = self._metrics(yva_np, val_probs)
        print(f"  [{self.target}] val_auc={metrics['val_auc']:.3f}  "
              f"val_ap={metrics['val_ap']:.3f}  "
              f"prec@10%={metrics['val_prec10']:.2%}")
        return metrics

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame, infer_batch: int = 8192) -> np.ndarray:
        import torch
        Xin = self._clean(X[self._feat_cols]).fillna(0.0).values.astype(np.float32)
        Xin = self._scaler.transform(Xin).astype(np.float32)
        Xin = np.clip(Xin, -self._scaled_clip, self._scaled_clip)
        dev = self._device or "cpu"
        self._net.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(Xin), infer_batch):
                Xt = torch.from_numpy(Xin[i: i + infer_batch]).to(dev)
                chunks.append(torch.sigmoid(self._net(Xt)).cpu().numpy())
        return np.concatenate(chunks)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
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
            "scaled_clip":           self._scaled_clip,
            "seed":                  self.seed,
            "hparams": dict(
                k=self.k, hidden_dim=self.hidden_dim,
                n_layers=self.n_layers, dropout=self.dropout,
            ),
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "TabMModel":
        import torch
        with open(path, "rb") as fh:
            p = pickle.load(fh)
        hp = p.get("hparams", {})
        m = cls(
            target     = p["target"],
            k          = hp.get("k", 8),
            hidden_dim = hp.get("hidden_dim", 256),
            n_layers   = hp.get("n_layers", 3),
            dropout    = hp.get("dropout", 0.15),
            seed       = p.get("seed", 42),
        )
        m._feat_cols              = p["feat_cols"]
        m._base_rate              = p["base_rate"]
        m._scaler                 = p["scaler"]
        m._feature_cols_override  = p.get("feature_cols_override")
        m._scaled_clip            = p.get("scaled_clip", 100.0)
        m._device                 = "cuda" if torch.cuda.is_available() else "cpu"

        n_feat = len(m._feat_cols)
        m._net = _build_net(n_feat, m.hidden_dim, m.n_layers,
                            m.k, m.dropout).to(m._device)
        m._net.load_state_dict(p["state_dict"])
        m._net.eval()
        return m

    @property
    def base_rate(self) -> float:
        return self._base_rate

    @property
    def feat_cols(self) -> list[str]:
        return self._feat_cols
