"""Integrated-gradients explanations for the ECG network (shared by the web app and the study)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .nets import TORCH

if TORCH:
    import torch


def load_nets(cache: Path, cond: str, width: int, device: str = "cpu"):
    """Load every saved seed of `cond` ('ecg_cnn' or 'fusion_e2e'): list of (net, mean, std) with mean/std as (12,) tensors."""
    from .nets import ECGNet
    out = []
    for f in sorted(Path(cache).glob(f"{cond}_seed*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        net = ECGNet(tab_dim=2 if ck["tab"] else 0, width=width)
        net.load_state_dict(ck["state"])
        net.eval().to(device)
        out.append((net, torch.as_tensor(np.asarray(ck["mean"]), dtype=torch.float32, device=device),
                    torch.as_tensor(np.asarray(ck["std"]), dtype=torch.float32, device=device)))
    return out


def ig_batch(net, x, steps: int = 32):
    """Integrated gradients of the logit of an ECG-only network.

    x: normalised signals (B, T, 12). Baseline = zeros, i.e. the training-set mean ECG. Midpoint rule with `steps` steps.
    Returns attributions (B, T, 12) and the completeness gap |sum(attr) - (f(x) - f(baseline))| per ECG.
    """
    B, T, C = x.shape
    a = ((torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps).view(1, steps, 1, 1)
    xs = (a * x[:, None]).reshape(B * steps, T, C).requires_grad_(True)
    with torch.enable_grad():
        net(xs.transpose(1, 2)).sum().backward()
    attr = (xs.grad.view(B, steps, T, C).mean(1) * x).detach()
    with torch.no_grad():
        f1 = net(x.transpose(1, 2))
        f0 = net(torch.zeros_like(x).transpose(1, 2))
    gap = (attr.sum((1, 2)) - (f1 - f0)).abs()
    return attr, gap
