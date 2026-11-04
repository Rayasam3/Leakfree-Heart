import numpy as np

from leakfree_heart.uncertainty import (aurc, calibrate_conformal, decisions_from_sets, max_coverage_at_accuracy,
                                        prediction_sets, risk_coverage)


def _sim(n, rng):
    p = rng.beta(0.6, 0.6, n)                  # true P(y=1|x); a perfectly calibrated "model"
    y = (rng.random(n) < p).astype(int)
    return p, y


def test_conformal_class_conditional_coverage():
    rng = np.random.default_rng(1)
    pc, yc = _sim(4000, rng)
    pt, yt = _sim(20000, rng)
    cal = calibrate_conformal(pc, yc)
    for alpha in (0.05, 0.1, 0.2):
        sets = prediction_sets(pt, cal.thresholds(alpha, mondrian=True))
        for k in (0, 1):
            cov = sets[yt == k, k].mean()
            assert cov >= 1 - alpha - 0.02, (alpha, k, cov)


def test_decisions_defer_on_ambiguous_sets():
    sets = np.array([[True, False], [False, True], [True, True], [False, False]])
    assert decisions_from_sets(sets).tolist() == [0, 1, -1, -1]


def test_selective_accuracy_improves_with_deferral():
    rng = np.random.default_rng(2)
    p, y = _sim(5000, rng)
    cov, acc = risk_coverage(y, p)
    assert acc[int(0.5 * len(acc))] > acc[-1] + 0.05
    assert aurc(y, p) < 1 - acc[-1]
    assert 0 <= max_coverage_at_accuracy(y, p, 0.9) <= 1
