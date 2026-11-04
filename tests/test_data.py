import numpy as np
import pandas as pd
import pytest

from leakfree_heart.config import load_config, path
from leakfree_heart.data import (FEATURES, deduplicate, duplicate_stats, kaggle_label_direction, kaggle_to_canonical,
                                 load_kaggle_raw, uci_to_canonical, UCI_RAW_COLUMNS)


def test_uci_conversion_recodes_and_nulls():
    raw = pd.DataFrame([
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],      # typical angina, fixed defect, no disease
        [67, 1, 4, 160, 0, 0, 2, 108, 1, 1.5, 2, np.nan, 3, 2],   # asymptomatic, chol 0 -> missing, disease
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 1, 0, 7, 1],      # non-anginal, reversible
    ], columns=UCI_RAW_COLUMNS)
    d = uci_to_canonical(raw, "cleveland")
    assert d.cp.tolist() == [0, 3, 2]
    assert d.slope.tolist() == [2, 1, 0]
    assert d.thal.tolist() == [2, 1, 3]
    assert np.isnan(d.loc[1, "chol"])
    assert d.target.tolist() == [0, 1, 1]
    assert list(d.columns[:13]) == FEATURES


def test_label_direction_detects_inverted_labels():
    rng = np.random.default_rng(0)
    n = 400
    disease = rng.integers(0, 2, n)
    df = pd.DataFrame({"exang": rng.binomial(1, 0.15 + 0.4 * disease), "oldpeak": rng.gamma(2, 0.4 + 0.5 * disease),
                       "ca": rng.binomial(3, 0.1 + 0.25 * disease)})
    df["target"] = disease
    assert kaggle_label_direction(df) == 1
    df["target"] = 1 - disease
    assert kaggle_label_direction(df) == -1


KAGGLE = path(load_config(), "data_raw") / "kaggle_heart.csv"


@pytest.mark.skipif(not KAGGLE.exists(), reason="Kaggle CSV not present")
def test_kaggle_copy_has_302_unique_patients_and_is_recoded():
    k = load_kaggle_raw(path(load_config(), "data_raw"))
    st = duplicate_stats(k)
    assert st["n_rows"] == 1025 and st["n_unique"] == 302
    c = kaggle_to_canonical(k)
    cd, removed = deduplicate(c)
    assert len(cd) == 302 and removed == 723
    # disease must now be the group with more exercise angina, ST depression and vessels
    g = cd.groupby("target")[["exang", "oldpeak", "ca"]].mean()
    assert (g.loc[1] > g.loc[0]).all()
    assert cd.thal.dropna().isin([1, 2, 3]).all() and cd.ca.dropna().between(0, 3).all()
    assert cd.isna().sum().sum() == 6          # 4 null ca + 2 null thal, as in the UCI Cleveland file
