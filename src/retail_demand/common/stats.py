"""Forecast metrics and the statistical tests used to compare forecasts.

Forecast errors are autocorrelated (a model that is wrong this hour is usually wrong the
next), so ordinary paired tests overstate significance. Comparisons therefore use:
  - the Diebold-Mariano test with a Newey-West (HAC) variance and the Harvey-Leybourne-
    Newbold small-sample correction, and
  - a moving-block bootstrap for confidence intervals, with blocks of whole weeks.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------- point metrics

def mae(y, f):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


def rmse(y, f):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2)))


def wape(y, f):
    """Weighted absolute percentage error: total absolute error / total demand."""
    y = np.asarray(y, float)
    return float(np.sum(np.abs(y - np.asarray(f, float))) / np.sum(np.abs(y)))


def bias(y, f):
    """Mean forecast minus mean actual, as a share of mean actual (+ = over-forecast)."""
    y = np.asarray(y, float)
    return float((np.mean(f) - np.mean(y)) / np.mean(y))


def mase(y, f, scale):
    """MAE scaled by the in-sample MAE of the seasonal-naive forecast (Hyndman & Koehler 2006)."""
    return mae(y, f) / scale


def pinball(y, q, alpha):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def point_metrics(y, f, mase_scale=None) -> dict:
    out = {"mae": mae(y, f), "rmse": rmse(y, f), "wape": wape(y, f), "bias": bias(y, f), "n": int(len(y))}
    if mase_scale:
        out["mase"] = mase(y, f, mase_scale)
    return out


# ------------------------------------------------------------------- comparison tests

def newey_west_var(d: np.ndarray, lag: int) -> float:
    """Long-run variance of the mean of d (Bartlett kernel)."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    lag = min(lag, n - 1)
    gamma0 = np.dot(d, d) / n
    s = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * np.dot(d[k:], d[:-k]) / n
    return max(s, 1e-12) / n


def diebold_mariano(e1, e2, h: int = 1, loss: str = "abs") -> dict:
    """H0: equal expected loss. Positive statistic = forecast 1 has the LARGER loss.

    h is the forecast horizon; the HAC lag is h-1 (errors of h-step forecasts are MA(h-1)
    under optimality), with the HLN small-sample correction and a t(n-1) reference."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    l1, l2 = (np.abs(e1), np.abs(e2)) if loss == "abs" else (e1 ** 2, e2 ** 2)
    d = l1 - l2
    n = len(d)
    var = newey_west_var(d, max(h - 1, 0))
    dm = np.mean(d) / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if n > 2 * h else 1.0
    stat = dm * hln
    p = 2 * sps.t.sf(abs(stat), df=n - 1)
    return {"mean_loss_diff": float(np.mean(d)), "statistic": float(stat), "p": float(p), "n": int(n), "h": int(h)}


def block_bootstrap(fn, n: int, block: int, n_boot: int = 1000, seed: int = 0) -> dict:
    """Moving-block bootstrap of a statistic fn(index_array) over a time-ordered sample."""
    rng = np.random.default_rng(seed)
    point = float(fn(np.arange(n)))
    n_blocks = int(np.ceil(n / block))
    starts_max = max(n - block, 0)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        draws[b] = fn(idx)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"value": point, "ci": [float(lo), float(hi)], "n_boot": n_boot, "block": block}


def coverage_test(y, lo, hi, nominal: float) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    inside = (y >= lo) & (y <= hi)
    k, n = int(inside.sum()), int(len(inside))
    bt = sps.binomtest(k, n, nominal)
    return {"coverage": k / n, "nominal": nominal, "n": n, "p_binomial": float(bt.pvalue),
            "below": float(np.mean(y < lo)), "above": float(np.mean(y > hi))}


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (keys preserved)."""
    keys = [k for k, v in pvalues.items() if v is not None and np.isfinite(v)]
    order = sorted(keys, key=lambda k: pvalues[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b): a rank effect size in [-1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = sps.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)
