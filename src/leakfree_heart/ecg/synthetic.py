"""Tiny synthetic PTB-XL-shaped dataset for smoke tests (--demo). NEVER used for reported results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _beat(t, c, w, a):
    return a * np.exp(-((t - c) ** 2) / (2 * w ** 2))


def make_synthetic_ptbxl(root: Path, n_records: int = 400, seed: int = 0) -> Path:
    import wfdb

    rng = np.random.default_rng(seed)
    root = Path(root)
    n_pat = int(n_records * 0.85)
    rows = []
    t = np.arange(1000) / 100.0
    for i in range(n_records):
        pid = int(rng.integers(0, n_pat))
        abnormal = rng.random() < 0.55
        cls = str(rng.choice(["MI", "STTC", "CD", "HYP"])) if abnormal else "NORM"
        hr = rng.uniform(55, 100)
        age = int(rng.integers(20, 90))
        sig = np.zeros((1000, 12))
        for beat_t in np.arange(0.3, 10, 60 / hr):
            for li in range(12):
                st = 0.25 * (1 if cls in ("MI", "STTC") else 0) * (1 if li in (6, 7, 8) else 0.3)
                wide = 0.04 if cls == "CD" else 0.02
                amp = 1.0 + (0.6 if cls == "HYP" else 0) + 0.1 * rng.normal()
                sig[:, li] += _beat(t, beat_t, wide, amp) + _beat(t, beat_t + 0.25, 0.06, 0.2 + st) \
                    + _beat(t, beat_t - 0.15, 0.03, 0.1)
        sig += 0.03 * rng.normal(size=sig.shape)
        rel = f"records100/{i // 1000:05d}/{i:05d}_lr"
        out = root / Path(rel).parent
        out.mkdir(parents=True, exist_ok=True)
        wfdb.wrsamp(Path(rel).name, fs=100, units=["mV"] * 12, sig_name=LEADS, p_signal=sig,
                    fmt=["16"] * 12, write_dir=str(out))
        rows.append(dict(ecg_id=i + 1, patient_id=pid, age=age if age < 90 else 300, sex=int(rng.integers(0, 2)),
                         height=np.nan, weight=np.nan, scp_codes=str({cls: 100.0}),
                         strat_fold=int(rng.integers(1, 11)), filename_lr=rel, filename_hr=rel.replace("100", "500")))
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    scp = pd.DataFrame({"code": ["NORM", "MI", "STTC", "CD", "HYP"], "diagnostic": [1] * 5,
                        "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP"]})
    scp.to_csv(root / "scp_statements.csv", index=False)
    return root
