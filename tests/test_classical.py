"""Classical benchmark: the week-ahead forecasts must equal the model's own forecasts and use no
data recorded in the 167 hours before the hour they predict."""

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from retail_demand import classical  # noqa: E402


def _series(n_weeks=14, seed=0, missing=0.05):
    rng = np.random.default_rng(seed)
    n = 168 * n_weeks
    how = np.arange(n) % 168
    y = 50 + 20 * np.sin(2 * np.pi * how / 24) + 5 * np.sin(2 * np.pi * how / 168) + rng.normal(0, 3, n) + np.cumsum(rng.normal(0, 0.05, n))
    y[rng.random(n) < missing] = np.nan
    return y, how


def test_fourier_design():
    X = classical.fourier(np.arange(336) % 168, K=3)
    assert X.shape == (336, 7) and np.allclose(X[:, 0], 1)
    assert np.allclose(X[:168], X[168:])                              # weekly period


@pytest.mark.parametrize("spec", [dict(order=(2, 0, 1)), dict(order=(2, 1, 1))])
def test_sarimax_week_ahead_matches_direct_forecast(spec):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y, how = _series()
    X = classical.fourier(how, 4)
    fit_end = 168 * 10
    out, _ = classical.sarimax_week_ahead(y, X, spec, fit_end)
    res = SARIMAX(y[:fit_end], exog=X[:fit_end], trend="n", **spec).fit(disp=False, maxiter=200)
    s = 168 * 12 + 5
    direct = res.apply(y[:s + 1], exog=X[:s + 1]).forecast(168, exog=X[s + 1:s + 169])[-1]
    assert out[s + 168] == pytest.approx(direct, abs=1e-8)


def test_sarimax_week_ahead_ignores_the_last_167_hours():
    y, how = _series()
    X = classical.fourier(how, 4)
    t = 168 * 12 + 30
    out, _ = classical.sarimax_week_ahead(y, X, dict(order=(1, 0, 1)), 168 * 10)
    y2 = y.copy()
    y2[t - 167:] = y2[t - 167:] * 3 + 40                              # change everything after t - 168
    out2, _ = classical.sarimax_week_ahead(y2, X, dict(order=(1, 0, 1)), 168 * 10)
    assert out[t] == pytest.approx(out2[t], abs=1e-9)
    y3 = y.copy()
    y3[t - 168] = (y3[t - 168] if np.isfinite(y3[t - 168]) else 50) + 500   # exactly one week before: allowed, must matter
    out3, _ = classical.sarimax_week_ahead(y3, X, dict(order=(1, 0, 1)), 168 * 10)
    assert out[t] != pytest.approx(out3[t], abs=1e-6)


@pytest.mark.parametrize("spec", [dict(trend=None), dict(trend="add", damped_trend=True)])
def test_holt_winters_week_ahead_matches_direct_forecast_and_is_week_ahead(spec):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y, _ = _series(missing=0)
    fit_end = 168 * 10
    out, _ = classical.holt_winters_week_ahead(y, spec, fit_end)
    p = ExponentialSmoothing(y[:fit_end], seasonal="add", seasonal_periods=168, initialization_method="estimated", **spec).fit().params
    known = dict(initialization_method="known", initial_level=p["initial_level"], initial_seasonal=p["initial_seasons"])
    fixed = dict(smoothing_level=p["smoothing_level"], smoothing_seasonal=p["smoothing_seasonal"], optimized=False)
    if spec["trend"]:
        known["initial_trend"] = p["initial_trend"]
        fixed.update(smoothing_trend=p["smoothing_trend"], damping_trend=p["damping_trend"])
    s = 168 * 12 + 5
    direct = ExponentialSmoothing(y[:s + 1], seasonal="add", seasonal_periods=168, **spec, **known).fit(**fixed).forecast(168)[-1]
    assert out[s + 168] == pytest.approx(direct, abs=1e-8)
    y2 = y.copy()
    y2[s + 1:] += 100
    out2, _ = classical.holt_winters_week_ahead(y2, spec, fit_end)
    assert out[s + 168] == pytest.approx(out2[s + 168], abs=1e-9)
