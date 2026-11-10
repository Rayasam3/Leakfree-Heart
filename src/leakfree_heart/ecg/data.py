"""PTB-XL loading, labels, patient-aware splits and the patient-leakage audit."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ABNORMAL = ["MI", "STTC", "CD", "HYP"]          # diagnostic superclasses other than NORM
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100                                        # Hz (records100)


def load_metadata(root: Path, min_likelihood: float = 50.0) -> pd.DataFrame:
    """One row per ECG with: patient_id, age, sex, fold, label (1 = abnormal, 0 = normal) and the superclass set.

    Label rule (documented in the paper): superclasses come from SCP diagnostic codes with likelihood >= min_likelihood.
    NORMAL = the only superclass is NORM.  ABNORMAL = at least one of MI, STTC, CD, HYP.
    ECGs with no diagnostic superclass are excluded.
    """
    root = Path(root)
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp["diagnostic"] == 1]["diagnostic_class"].to_dict()

    def supers(codes) -> set:
        d = ast.literal_eval(codes) if isinstance(codes, str) else dict(codes)
        return {diag[c] for c, lk in d.items() if c in diag and float(lk) >= min_likelihood}

    meta["superclasses"] = meta["scp_codes"].apply(supers)
    meta["label"] = meta["superclasses"].apply(
        lambda s: np.nan if not s else (0 if s == {"NORM"} else 1))
    meta["age_years"] = meta["age"].astype(float).where(meta["age"] < 120, 90.0)   # PTB-XL encodes age > 89 as 300
    meta["sex_f"] = meta["sex"].astype(int)                                          # 0 = male, 1 = female
    meta["fold"] = meta["strat_fold"].astype(int)
    meta["n_ecg_patient"] = meta.groupby("patient_id")["patient_id"].transform("size")
    meta = meta[meta["label"].notna()].copy()
    meta["label"] = meta["label"].astype(int)
    return meta


def load_signals(root: Path, meta: pd.DataFrame, cache: Path | None = None) -> np.ndarray:
    """Return signals as float16 array (N, 1000, 12) in the row order of `meta` (cached as .npy)."""
    import wfdb

    root = Path(root)
    if cache is not None and Path(cache).exists():
        arr = np.load(cache, mmap_mode="r")
        if arr.shape[0] == len(meta):
            return arr
    X = np.zeros((len(meta), 1000, 12), dtype=np.float16)
    for i, name in enumerate(meta["filename_lr"].tolist()):
        sig, _ = wfdb.rdsamp(str(root / name))
        sig = np.nan_to_num(sig.astype(np.float32))
        X[i, : min(1000, len(sig))] = sig[:1000]
        if (i + 1) % 4000 == 0:
            print(f"   read {i + 1}/{len(meta)} ECGs", flush=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    return X


def official_split(meta: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """Patient-wise split from the official PTB-XL stratified folds (a patient is never in two folds).

    train: folds 1-7   |  stop: fold 8 (early stopping)   |  cal_a: half of fold 9 (stacking / temperature)
    cal_b: other half of fold 9 (conformal calibration, split by patient)   |  test: fold 10
    """
    fold = meta["fold"].to_numpy()
    out = {"train": np.where(fold <= 7)[0], "stop": np.where(fold == 8)[0], "test": np.where(fold == 10)[0]}
    f9 = np.where(fold == 9)[0]
    pats = np.unique(meta["patient_id"].to_numpy()[f9])
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    half = set(pats[: len(pats) // 2].tolist())
    in_a = np.array([p in half for p in meta["patient_id"].to_numpy()[f9]])
    out["cal_a"], out["cal_b"] = f9[in_a], f9[~in_a]
    return out


def random_record_split(meta: pd.DataFrame, sizes: dict[str, int], seed: int = 42) -> dict[str, np.ndarray]:
    """Record-wise random split (ignores patients) with the same part sizes as the official split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(meta))
    out, pos = {}, 0
    for k in ("train", "stop", "cal_a", "cal_b", "test"):
        out[k] = np.sort(idx[pos: pos + sizes[k]])
        pos += sizes[k]
    return out


def patient_overlap(meta: pd.DataFrame, split: dict[str, np.ndarray]) -> dict:
    """Fraction of test ECGs whose patient also has an ECG in the training set."""
    pid = meta["patient_id"].to_numpy()
    tr = set(pid[split["train"]].tolist())
    te = pid[split["test"]]
    shared = np.array([p in tr for p in te])
    return {"n_test": int(len(te)), "n_test_patient_in_train": int(shared.sum()), "fraction": float(shared.mean())}


def dataset_summary(meta: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("ECGs (with a diagnostic label)", len(meta)),
        ("Patients", meta["patient_id"].nunique()),
        ("ECGs per patient (max)", int(meta["n_ecg_patient"].max())),
        ("Patients with >1 ECG", int((meta.groupby("patient_id").size() > 1).sum())),
        ("Normal ECGs", int((meta["label"] == 0).sum())),
        ("Abnormal ECGs", int((meta["label"] == 1).sum())),
        ("Abnormal prevalence", round(float(meta["label"].mean()), 4)),
        ("Female fraction", round(float(meta["sex_f"].mean()), 4)),
        ("Age median (years)", float(meta["age_years"].median())),
    ]
    for c in ABNORMAL:
        rows.append((f"ECGs with superclass {c}", int(meta["superclasses"].apply(lambda s, c=c: c in s).sum())))
    return pd.DataFrame(rows, columns=["item", "value"])
