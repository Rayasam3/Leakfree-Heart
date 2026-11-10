"""scikit-learn compatible PyTorch tabular classifiers (dense network and 1-D CNN).

Both expose fit / predict_proba so they plug into repeated CV, calibration and conformal prediction.
Model selection uses an *internal* validation split carved from the training data - never the test fold.
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def resolve_device(device: str = "auto") -> str:
    if not TORCH_AVAILABLE:
        return "cpu"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _build_dense(n_in: int, hidden: tuple[int, ...], dropout: float):
    layers, d = [], n_in
    for h in hidden:                       # base paper: 128-64-32-64-32, ReLU, dropout 0.2
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class _Conv1DNet(nn.Module if TORCH_AVAILABLE else object):
    """A genuine convolutional network over the feature sequence (13 x 1 channel)."""

    def __init__(self, n_in: int, dropout: float = 0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):                  # x: (B, F)
        z = self.conv(x.unsqueeze(1)).squeeze(-1)
        return self.head(z)


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, arch: str = "dense", hidden=(128, 64, 32, 64, 32), dropout: float = 0.2,
                 epochs: int = 300, patience: int | None = 40, lr: float = 1e-3, batch_size: int = 32,
                 weight_decay: float = 1e-4, val_fraction: float = 0.15, select_by: str = "val_loss",
                 device: str = "auto", random_state: int = 0, verbose: bool = False):
        self.arch = arch
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.select_by = select_by         # "val_loss" (default) or "val_acc" (base-paper protocol)
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def _net(self, n_in):
        if self.arch == "dense":
            return _build_dense(n_in, tuple(self.hidden), self.dropout)
        if self.arch == "conv1d":
            return _Conv1DNet(n_in, self.dropout)
        raise ValueError(self.arch)

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. See SETUP_AND_RUN_GUIDE.md")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        dev = resolve_device(self.device)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        strat = y if min(np.bincount(y.astype(int))) >= 2 else None
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.val_fraction,
                                              stratify=strat, random_state=self.random_state)
        Xtr_t, ytr_t = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
        Xva_t, yva_t = torch.tensor(Xva, device=dev), torch.tensor(yva, device=dev)

        net = self._net(X.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        g = torch.Generator(device="cpu").manual_seed(self.random_state)

        best_state, best_key, bad = None, None, 0
        hist = {"train_loss": [], "val_loss": [], "val_acc": []}
        n = len(Xtr_t)
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(n, generator=g).to(dev)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:           # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss = lossf(net(Xtr_t[idx]).squeeze(1), ytr_t[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                logit = net(Xva_t).squeeze(1)
                vloss = lossf(logit, yva_t).item()
                vacc = ((logit > 0).float() == yva_t).float().mean().item()
            hist["train_loss"].append(tot / n)
            hist["val_loss"].append(vloss)
            hist["val_acc"].append(vacc)
            key = (-vloss,) if self.select_by == "val_loss" else (vacc, -vloss)
            if best_key is None or key > best_key:
                best_key, best_state, bad = key, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if self.patience is not None and bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.device_ = dev
        self.history_ = hist
        self.best_epoch_ = int(np.argmin(hist["val_loss"])) if self.select_by == "val_loss" else int(np.argmax(hist["val_acc"]))
        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(X, device=self.device_)).squeeze(1)).cpu().numpy()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TabNetWrapper(ClassifierMixin, BaseEstimator):
    """Optional TabNet (pip install pytorch-tabnet). Same interface as above."""

    def __init__(self, max_epochs: int = 200, patience: int = 30, batch_size: int = 64,
                 val_fraction: float = 0.15, device: str = "auto", random_state: int = 0):
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.val_fraction, stratify=y,
                                              random_state=self.random_state)
        self.model_ = TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=self.random_state,
                                       device_name=resolve_device(self.device), verbose=0)
        self.model_.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=["logloss"],
                        max_epochs=self.max_epochs, patience=self.patience,
                        batch_size=self.batch_size, virtual_batch_size=min(32, self.batch_size))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
