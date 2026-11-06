"""1-D residual ECG network, optional clinical (age, sex) branch for fusion, and the training loop."""
from __future__ import annotations

import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except Exception:  # noqa: BLE001
    torch = nn = F = None
    TORCH = False


def device_name(pref: str = "auto") -> str:
    if not TORCH:
        return "cpu"
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (pref == "cuda" and torch.cuda.is_available()) else "cpu"


if TORCH:
    class ResBlock1d(nn.Module):
        def __init__(self, cin, cout, stride, k=7, drop=0.1):
            super().__init__()
            self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
            self.b1 = nn.BatchNorm1d(cout)
            self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
            self.b2 = nn.BatchNorm1d(cout)
            self.drop = nn.Dropout(drop)
            self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

        def forward(self, x):
            y = F.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(self.drop(y)))
            return F.relu(y + (x if self.skip is None else self.skip(x)))

    class ECGNet(nn.Module):
        """ECG (B,12,T) [+ tabular (B,tab_dim)] -> abnormal-ECG logit.

        tab_dim = 0  -> ECG only.   tab_dim > 0 -> late fusion: ECG embedding (128) concatenated with a small MLP on age/sex.
        """

        def __init__(self, tab_dim: int = 0, width: int = 32, emb: int = 128, drop: float = 0.2):
            super().__init__()
            w = width
            self.stem = nn.Sequential(nn.Conv1d(12, w, 15, 2, 7, bias=False), nn.BatchNorm1d(w), nn.ReLU())
            self.body = nn.Sequential(
                ResBlock1d(w, w, 1), ResBlock1d(w, 2 * w, 2), ResBlock1d(2 * w, 2 * w, 1),
                ResBlock1d(2 * w, 4 * w, 2), ResBlock1d(4 * w, 4 * w, 1), ResBlock1d(4 * w, 8 * w, 2))
            self.proj = nn.Sequential(nn.Linear(16 * w, emb), nn.ReLU(), nn.Dropout(drop))
            self.tab_dim = tab_dim
            if tab_dim:
                self.tab = nn.Sequential(nn.Linear(tab_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
                self.head = nn.Linear(emb + 32, 1)
            else:
                self.head = nn.Linear(emb, 1)

        def embed(self, x):
            z = self.body(self.stem(x))
            return self.proj(torch.cat([z.mean(-1), z.amax(-1)], 1))

        def forward(self, x, t=None):
            h = self.embed(x)
            if self.tab_dim:
                h = torch.cat([h, self.tab(t)], 1)
            return self.head(h).squeeze(-1)


def _tens(a, dev, dtype=None):
    t = torch.as_tensor(np.asarray(a))
    return t.to(dev) if dtype is None else t.to(dev, dtype=dtype)


def train_net(Xtr, ytr, Xva, yva, ttr=None, tva=None, *, seed=0, epochs=30, batch_size=128, lr=2e-3,
              patience=6, device="auto", width=32, verbose=True):
    """Train on (Xtr,ytr), early-stop on validation AUC. X: (N,T,12) float16/32. Returns (model, mean, std)."""
    from sklearn.metrics import roc_auc_score
    dev = device_name(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tab_dim = 0 if ttr is None else ttr.shape[1]
    model = ECGNet(tab_dim=tab_dim, width=width).to(dev)
    mean = np.asarray(Xtr[: min(len(Xtr), 4000)], dtype=np.float32).mean((0, 1))
    std = np.asarray(Xtr[: min(len(Xtr), 4000)], dtype=np.float32).std((0, 1)) + 1e-6
    Xt = _tens(Xtr, dev, torch.float16)
    yt = _tens(ytr, dev, torch.float32)
    tt = None if ttr is None else _tens(ttr, dev, torch.float32)
    mean_t, std_t = _tens(mean, dev, torch.float32), _tens(std, dev, torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = max(1, int(np.ceil(len(Xt) / batch_size))) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    best, best_state, bad = -1.0, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(dev)
        tot = 0.0
        for s in range(0, len(Xt), batch_size):
            b = perm[s: s + batch_size]
            x = ((Xt[b].float() - mean_t) / std_t)
            # light augmentation: amplitude scaling, baseline-free noise, random circular time shift
            x = x * (1 + 0.1 * torch.randn(len(b), 1, 1, device=dev)) + 0.02 * torch.randn_like(x)
            sh = int(torch.randint(-40, 41, (1,), generator=g))
            x = torch.roll(x, sh, dims=1)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                out = model(x.transpose(1, 2), None if tt is None else tt[b])
                loss = F.binary_cross_entropy_with_logits(out.float(), yt[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(b)
        p = predict_logits(model, Xva, tva, mean, std, dev)
        auc = roc_auc_score(yva, p) if len(np.unique(yva)) == 2 else 0.5
        if verbose:
            print(f"      epoch {ep + 1:02d}/{epochs}  train loss {tot / len(Xt):.4f}  val AUC {auc:.4f}", flush=True)
        if auc > best:
            best, bad = auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std


@torch.no_grad() if TORCH else (lambda f: f)
def predict_logits(model, X, t=None, mean=None, std=None, device="auto", batch_size=512):
    dev = device_name(device) if isinstance(device, str) else device
    model.eval()
    out = []
    m, s = _tens(mean, dev, torch.float32), _tens(std, dev, torch.float32)
    for i in range(0, len(X), batch_size):
        x = (_tens(X[i: i + batch_size], dev, torch.float32) - m) / s
        tb = None if t is None else _tens(t[i: i + batch_size], dev, torch.float32)
        out.append(model(x.transpose(1, 2), tb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
