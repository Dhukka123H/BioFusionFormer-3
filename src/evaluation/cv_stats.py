"""
Statistics for repeated k-fold cross-validation (numpy/scipy only; no torch).

Scores from overlapping CV folds are NOT independent: every training set shares ~80% of its
patients with the others. The naive interval mean +/- 1.96*SD/sqrt(K) therefore shrinks as the number
of folds K grows although no new patients are added. The Nadeau-Bengio correction adds a term
n_test/n_train that does not depend on K:

    Var_corrected = (1/K + n_test/n_train) * s^2

Consequence: with N=121 (n_train~97, n_test~24) going from 10 to 25 folds changes the corrected
interval only marginally, because the n_test/n_train term (~0.25) dominates 1/K (0.10 -> 0.04).
"""
import math

import numpy as np
from scipy import stats


def nadeau_bengio_t(scores_a, scores_b, n_train=96, n_test=25):
    """Corrected resampled paired t-test. Returns (t, p, mean_diff, se)."""
    diffs = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    K = len(diffs)
    mean_diff = float(diffs.mean())
    var_d = float(diffs.var(ddof=1))
    se = math.sqrt(max((1.0 / K + n_test / n_train) * var_d, 1e-12))
    t = mean_diff / se
    p = 2.0 * float(stats.t.sf(abs(t), df=K - 1))
    return float(t), p, mean_diff, se


def nb_interval(scores, n_train, n_test, level=0.95):
    """Corrected t-interval for the mean of K overlapping CV scores. Returns (mean, lo, hi, se)."""
    x = np.asarray(scores, dtype=float)
    K = len(x)
    m = float(x.mean())
    se = math.sqrt(max((1.0 / K + n_test / n_train) * float(x.var(ddof=1)), 1e-12))
    tcrit = float(stats.t.ppf(0.5 + level / 2.0, K - 1))
    return m, m - tcrit * se, m + tcrit * se, se


def naive_interval(scores, level=0.95):
    """Anti-conservative interval (treats folds as independent). For comparison only."""
    x = np.asarray(scores, dtype=float)
    m = float(x.mean())
    half = float(stats.norm.ppf(0.5 + level / 2.0)) * float(x.std(ddof=1)) / math.sqrt(len(x))
    return m, m - half, m + half
