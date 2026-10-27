"""Tests for the CPI forecast and oil scenarios, using simulated data."""
import numpy as np
import pandas as pd

from emsp.modeling.forecast import forecast_cpi, oil_surprise_pct
from emsp.modeling.irf import price_level_response
from emsp.modeling.var_model import (
    ENDOGENOUS,
    EXCHANGE_RATE,
    fit_var,
    make_exog,
    prepare_var_data,
)
from emsp.processing.diagnostics import log_diff


def make_fake_levels() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    index = pd.date_range("2000-12-01", "2026-08-01", freq="MS")
    n = len(index)

    def path(start, drift, volatility):
        return start * np.exp(np.cumsum(rng.normal(drift, volatility, n)) / 100)

    return pd.DataFrame({
        "cpi_index": path(100, 0.8, 0.5),
        "etb_per_usd": path(10, 0.5, 1.0),
        "brent_usd_bbl": path(30, 0.0, 5.0),
        "urea_usd_mt": path(200, 0.0, 5.0),
    }, index=index)


def fit_fake_model():
    levels = make_fake_levels()
    endog, exog = prepare_var_data(levels)
    return levels, exog, fit_var(endog, exog, lags=1)


def test_forecast_starts_the_month_after_the_data():
    levels, exog, results = fit_fake_model()
    forecast = forecast_cpi(results, levels, exog.columns, horizon=12)

    assert len(forecast) == 12
    assert forecast.index[0] == pd.Timestamp("2026-09-01")


def test_first_forecast_month_matches_hand_calculation():
    levels, exog, results = fit_fake_model()
    forecast = forecast_cpi(results, levels, exog.columns, horizon=3, monthly_depreciation_pct=2.0)

    # Rebuild the first month by hand from the estimated coefficients
    changes = log_diff(levels[ENDOGENOUS + [EXCHANGE_RATE]])
    month = pd.Timestamp("2026-09-01")
    fx_path = pd.concat([changes[EXCHANGE_RATE], pd.Series([2.0], index=[month])])
    inputs = make_exog(fx_path).loc[month, list(exog.columns)]
    coef = results.params["cpi_index"]

    by_hand = (
        coef["const"]
        + sum(coef[col] * inputs[col] for col in exog.columns)
        + sum(coef[f"L1.{var}"] * changes[var].iloc[-1] for var in ENDOGENOUS)
    )
    assert abs(by_hand - forecast["monthly_inflation_pct"].iloc[0]) < 1e-9


def test_zero_oil_shock_gives_the_baseline():
    levels, exog, results = fit_fake_model()
    baseline = forecast_cpi(results, levels, exog.columns, horizon=12)
    no_shock = forecast_cpi(results, levels, exog.columns, horizon=12, oil_shock_pct=0.0)
    assert np.allclose(baseline.to_numpy(), no_shock.to_numpy())


def test_oil_scenario_gap_equals_the_impulse_response():
    levels, exog, results = fit_fake_model()
    baseline = forecast_cpi(results, levels, exog.columns, horizon=12)
    shocked = forecast_cpi(results, levels, exog.columns, horizon=12, oil_shock_pct=25.0)

    gap = 100 * np.log(shocked["cpi_index"] / baseline["cpi_index"])
    expected = price_level_response(results, "brent_usd_bbl", "cpi_index", horizon=11, shock_pct=25.0)
    assert np.allclose(gap.to_numpy(), expected.to_numpy())
    assert abs(shocked["oil_change_pct"].iloc[0] - baseline["oil_change_pct"].iloc[0] - 25.0) < 1e-9


def test_no_surprise_when_oil_matches_the_baseline_forecast():
    levels, exog, results = fit_fake_model()
    baseline = forecast_cpi(results, levels, exog.columns, horizon=1)
    expected_price = levels["brent_usd_bbl"].iloc[-1] * np.exp(baseline["oil_change_pct"].iloc[0] / 100)

    assert abs(oil_surprise_pct(results, levels, exog.columns, expected_price)) < 1e-9