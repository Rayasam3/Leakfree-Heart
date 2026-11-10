"""Dataset loading, harmonisation and de-duplication.

Canonical coding used everywhere in this project (UCI conventions, 0-based):

    sex      0 female, 1 male
    cp       0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic
    fbs      1 if fasting blood sugar > 120 mg/dl
    restecg  0 normal, 1 ST-T abnormality, 2 probable/definite LV hypertrophy
    exang    1 exercise-induced angina
    slope    0 upsloping, 1 flat, 2 downsloping (peak-exercise ST segment)
    ca       0-3 major vessels coloured by fluoroscopy
    thal     1 normal, 2 fixed defect, 3 reversible defect
    target   1 = angiographic heart disease (UCI "num" > 0), 0 = none

The widely used Kaggle copy of the Cleveland data uses a *different* coding (inverted target,
reordered cp / slope / thal categories, nulls stored as ca=4 and thal=0).  `kaggle_to_canonical`
converts it.  Its 1025 rows contain only 302 unique patients.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET = "target"
SITES = ["cleveland", "hungarian", "switzerland", "va"]

UCI_RAW_COLUMNS = FEATURES + ["num"]
UCI_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
UCI_EXPECTED_ROWS = {"cleveland": 303, "hungarian": 294, "switzerland": 123, "va": 200}

# Human readable metadata, used by the API, the prompts and the React form.
FEATURE_INFO: dict[str, dict] = {
    "age": dict(label="Age", unit="years", kind="numeric", min=20, max=90, default=55,
                desc="Age in years"),
    "sex": dict(label="Sex", kind="categorical", options={0: "Female", 1: "Male"}, default=1,
                desc="Biological sex"),
    "cp": dict(label="Chest pain type", kind="categorical",
               options={0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
               default=0, desc="Type of chest pain"),
    "trestbps": dict(label="Resting blood pressure", unit="mmHg", kind="numeric", min=80, max=220,
                     default=130, desc="Resting systolic blood pressure on admission"),
    "chol": dict(label="Serum cholesterol", unit="mg/dL", kind="numeric", min=100, max=600,
                 default=240, desc="Serum cholesterol"),
    "fbs": dict(label="Fasting blood sugar > 120 mg/dL", kind="categorical",
                options={0: "No", 1: "Yes"}, default=0, desc="Fasting blood sugar above 120 mg/dL"),
    "restecg": dict(label="Resting ECG", kind="categorical",
                    options={0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"},
                    default=0, desc="Resting electrocardiographic result"),
    "thalach": dict(label="Maximum heart rate achieved", unit="bpm", kind="numeric", min=60, max=220,
                    default=150, desc="Maximum heart rate achieved during the exercise test"),
    "exang": dict(label="Exercise-induced angina", kind="categorical",
                  options={0: "No", 1: "Yes"}, default=0, desc="Angina provoked by exercise"),
    "oldpeak": dict(label="ST depression (exercise vs rest)", unit="mm", kind="numeric", min=0.0, max=7.0,
                    step=0.1, default=1.0, desc="ST depression induced by exercise relative to rest"),
    "slope": dict(label="Peak-exercise ST slope", kind="categorical",
                  options={0: "Upsloping", 1: "Flat", 2: "Downsloping"}, default=1,
                  desc="Slope of the peak exercise ST segment"),
    "ca": dict(label="Major vessels coloured by fluoroscopy", kind="numeric", min=0, max=3, step=1,
               default=0, desc="Number of major vessels (0-3) coloured by fluoroscopy"),
    "thal": dict(label="Thallium stress test", kind="categorical",
                 options={1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}, default=1,
                 desc="Thallium scintigraphy result"),
}


# --------------------------------------------------------------------------- UCI
def _read_uci_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=UCI_RAW_COLUMNS, na_values=["?", "-9", "-9.0"])


def uci_to_canonical(raw: pd.DataFrame, site: str) -> pd.DataFrame:
    """Convert a raw UCI 'processed' table to the canonical coding."""
    df = raw.copy()
    for c in UCI_RAW_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cp"] = df["cp"] - 1                      # 1..4 -> 0..3
    df["slope"] = df["slope"] - 1                # 1..3 -> 0..2
    df["thal"] = df["thal"].map({3.0: 1.0, 6.0: 2.0, 7.0: 3.0})
    # Physiologically impossible zeros are encoded missing values in several sites.
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df[TARGET] = (df["num"] > 0).astype(int)
    df["site"] = site
    return df[FEATURES + [TARGET, "site"]].reset_index(drop=True)


def load_uci_site(site: str, raw_dir: Path) -> pd.DataFrame:
    f = Path(raw_dir) / "uci" / UCI_FILES[site]
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run: python scripts/download_data.py")
    return uci_to_canonical(_read_uci_file(f), site)


def uci_available(raw_dir: Path) -> bool:
    return all((Path(raw_dir) / "uci" / f).exists() for f in UCI_FILES.values())


# ------------------------------------------------------------------------ Kaggle
KAGGLE_CP_MAP = {0: 3, 1: 1, 2: 2, 3: 0}        # Kaggle cp -> canonical cp
KAGGLE_THAL_MAP = {2: 1, 1: 2, 3: 3}            # Kaggle thal -> canonical thal (0 = null)


def kaggle_label_direction(df: pd.DataFrame, target: str = "target") -> int:
    """+1 if target==1 means disease, -1 if the labels are inverted.

    Uses three clinically unambiguous disease markers (exercise angina, ST depression, number of
    coloured vessels): their mean must be higher in the diseased group."""
    score = 0.0
    for c in ("exang", "oldpeak", "ca"):
        x = pd.to_numeric(df[c], errors="coerce")
        m1, m0 = x[df[target] == 1].mean(), x[df[target] == 0].mean()
        score += np.sign(m1 - m0)
    return 1 if score > 0 else -1


def kaggle_to_canonical(df: pd.DataFrame, site: str = "cleveland") -> pd.DataFrame:
    """Recode the Kaggle 'heart.csv' (1025 rows) into canonical UCI conventions (duplicates kept)."""
    out = df.copy()
    direction = kaggle_label_direction(out)
    out["cp"] = out["cp"].map(KAGGLE_CP_MAP)
    out["slope"] = 2 - out["slope"]
    out["thal"] = out["thal"].map(KAGGLE_THAL_MAP)          # 0 -> NaN
    out["ca"] = out["ca"].where(out["ca"] <= 3, np.nan)     # 4 -> NaN
    out[TARGET] = out[TARGET] if direction == 1 else 1 - out[TARGET]
    out["site"] = site
    return out[FEATURES + [TARGET, "site"]].reset_index(drop=True)


def load_kaggle_raw(raw_dir: Path) -> pd.DataFrame:
    f = Path(raw_dir) / "kaggle_heart.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run: python scripts/download_data.py --kaggle")
    return pd.read_csv(f)


# --------------------------------------------------------------------- top level
def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate patient records (features + label + site)."""
    n0 = len(df)
    out = df.drop_duplicates(subset=FEATURES + [TARGET, "site"]).reset_index(drop=True)
    return out, n0 - len(out)


def duplicate_stats(df: pd.DataFrame, cols: list[str] | None = None) -> dict:
    cols = cols or [c for c in df.columns if c != "site"]
    dup_mask = df.duplicated(subset=cols, keep="first")
    sizes = df.groupby(cols, dropna=False).size()
    return {
        "n_rows": int(len(df)),
        "n_unique": int(len(df) - dup_mask.sum()),
        "n_duplicate_rows": int(dup_mask.sum()),
        "duplicate_fraction": float(dup_mask.mean()),
        "n_groups_with_copies": int((sizes > 1).sum()),
        "max_copies": int(sizes.max()),
    }


def load_dataset(cfg: dict, raw_dir: Path, sites: list[str] | None = None,
                 dedup: bool | None = None) -> pd.DataFrame:
    """Load the canonical dataset according to cfg['data']['source'].

    Returns a DataFrame with FEATURES + target + site. When `dedup` (default: cfg) is true,
    exact duplicate patients are removed."""
    src = cfg["data"]["source"]
    dedup = cfg["data"]["deduplicate"] if dedup is None else dedup
    if src == "auto":
        src = "uci" if uci_available(raw_dir) else "kaggle_cleveland"
    if src == "uci":
        wanted = sites or SITES
        df = pd.concat([load_uci_site(s, raw_dir) for s in wanted], ignore_index=True)
    elif src == "kaggle_cleveland":
        df = kaggle_to_canonical(load_kaggle_raw(raw_dir))
        if sites and "cleveland" not in sites:
            raise FileNotFoundError("Only Cleveland is available (Kaggle copy). "
                                    "Run python scripts/download_data.py for the other sites.")
    else:
        raise ValueError(f"Unknown data.source: {src}")
    df.attrs["source"] = src
    if dedup:
        df, removed = deduplicate(df)
        df.attrs["source"] = src
        df.attrs["duplicates_removed"] = removed
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()
