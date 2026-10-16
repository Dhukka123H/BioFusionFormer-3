import math
import numpy as np
from src.evaluation.cv_stats import nadeau_bengio_t, nb_interval, naive_interval


def test_nb_wider_than_naive():
    rng = np.random.RandomState(0)
    x = 0.87 + 0.06 * rng.randn(25)
    m, lo, hi, se = nb_interval(x, 97, 24)
    nm, nlo, nhi = naive_interval(x)
    assert m == nm and (hi - lo) > 2 * (nhi - nlo)


def test_more_folds_barely_helps():
    """The corrected SE factor sqrt(1/K + n_test/n_train) changes by <12% between K=10 and K=25."""
    f = lambda K: math.sqrt(1 / K + 24 / 97)
    assert f(10) / f(25) < 1.12


def test_nb_t_matches_formula_and_is_symmetric():
    a = np.array([0.9, 0.85, 0.88, 0.91, 0.87])
    b = np.array([0.8, 0.82, 0.79, 0.85, 0.80])
    t, p, d, se = nadeau_bengio_t(a, b, n_train=96, n_test=24)
    t2, p2, d2, _ = nadeau_bengio_t(b, a, n_train=96, n_test=24)
    assert abs(t + t2) < 1e-9 and abs(p - p2) < 1e-12 and abs(d + d2) < 1e-12
    var = np.var(a - b, ddof=1)
    assert abs(se - math.sqrt((1 / 5 + 24 / 96) * var)) < 1e-12


def test_identical_scores_give_p_one():
    a = np.array([0.8, 0.9, 0.85, 0.7])
    t, p, d, se = nadeau_bengio_t(a, a)
    assert d == 0 and abs(p - 1.0) < 1e-6
