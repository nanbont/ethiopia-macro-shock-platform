"""Tests for the impulse response helpers, using simulated data."""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from emsp.modeling.irf import (
    bootstrap_price_level_response,
    price_level_response,
    simulate_from_residuals,
)


def make_fake_data():
    rng = np.random.default_rng(0)
    index = pd.date_range("2001-01-01", periods=200, freq="MS")
    endog = pd.DataFrame(rng.normal(size=(200, 2)), index=index, columns=["oil", "cpi"])
    exog = pd.DataFrame({"fx": rng.normal(size=200)}, index=index)
    return endog, exog


def test_shock_is_exactly_the_requested_size_on_impact():
    endog, exog = make_fake_data()
    results = VAR(endog, exog=exog).fit(1)
    response = price_level_response(results, "oil", "oil", horizon=12, shock_pct=10.0)
    assert abs(response.iloc[0] - 10.0) < 1e-9


def test_response_covers_every_month_of_the_horizon():
    endog, exog = make_fake_data()
    results = VAR(endog, exog=exog).fit(1)
    assert len(price_level_response(results, "oil", "cpi", horizon=12)) == 13


def test_simulation_with_original_residuals_reproduces_the_data():
    endog, exog = make_fake_data()
    for lags in [1, 2]:
        results = VAR(endog, exog=exog).fit(lags)
        rebuilt = simulate_from_residuals(results, endog, exog, results.resid.to_numpy())
        assert np.allclose(rebuilt.to_numpy(), endog.to_numpy())


def test_bootstrap_band_has_lower_below_upper():
    endog, exog = make_fake_data()
    results = VAR(endog, exog=exog).fit(1)
    band = bootstrap_price_level_response(
        results, endog, exog, "oil", "cpi", horizon=6, replications=50
    )
    assert list(band.columns) == ["lower", "estimate", "upper"]
    assert (band["lower"] <= band["upper"]).all()