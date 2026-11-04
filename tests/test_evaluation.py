import numpy as np

from leakfree_heart.evaluation import binary_metrics, corrected_resampled_ttest, ece_top_label, holm


def test_corrected_t_is_more_conservative_than_naive():
    from scipy import stats
    rng = np.random.default_rng(0)
    a = 0.85 + rng.normal(0, 0.02, 50)
    b = a - 0.01 + rng.normal(0, 0.02, 50)
    t, p = corrected_resampled_ttest(a, b, n_train=270, n_test=30)
    p_naive = stats.ttest_rel(a, b).pvalue
    assert p >= p_naive


def test_holm_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert all(0 <= x <= 1 for x in adj) and adj[0] <= adj[2] <= adj[1]


def test_metrics_and_ece():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_metrics(y, p)
    assert m["accuracy"] == 1 and m["auc"] == 1 and m["sensitivity"] == 1 and m["specificity"] == 1
    assert 0 <= ece_top_label(y, p) < 0.2
