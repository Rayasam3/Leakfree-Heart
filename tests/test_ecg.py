"""Smoke tests for the ECG branch on a tiny synthetic PTB-XL-shaped dataset (no download needed)."""
import numpy as np
import pytest

from leakfree_heart.ecg import data as D
from leakfree_heart.ecg.features import ecg_features
from leakfree_heart.ecg.synthetic import make_synthetic_ptbxl


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = make_synthetic_ptbxl(tmp_path_factory.mktemp("ptbxl"), 60, seed=1)
    meta = D.load_metadata(root)
    X = D.load_signals(root, meta)
    return root, meta, X


def test_labels_and_shapes(synth):
    _, meta, X = synth
    assert X.shape == (len(meta), 1000, 12)
    assert set(meta["label"].unique()) <= {0, 1}
    assert (meta["age_years"] <= 90).all()


def test_split_is_disjoint(synth):
    _, meta, _ = synth
    sp = D.official_split(meta)
    seen = set()
    for k, v in sp.items():
        assert not (seen & set(v.tolist())), k
        seen |= set(v.tolist())


def test_patient_overlap_zero_when_split_by_patient(synth):
    _, meta, _ = synth
    meta = meta.copy()
    meta["fold"] = (meta["patient_id"] % 10 + 1).astype(int)       # patient-wise folds, like real PTB-XL
    sp = D.official_split(meta)
    assert D.patient_overlap(meta, sp)["fraction"] == 0.0


def test_features(synth):
    _, _, X = synth
    f = ecg_features(X[:5])
    assert f.shape[0] == 5 and not f.isna().any().any()


def test_nets_forward():
    torch = pytest.importorskip("torch")
    from leakfree_heart.ecg.nets import ECGNet
    m = ECGNet(tab_dim=2)
    out = m(torch.randn(4, 12, 1000), torch.randn(4, 2))
    assert out.shape == (4,)
