"""VARX model: oil, urea and CPI as endogenous variables, the birr as an external input.

The birr/USD rate was set administratively by the National Bank of Ethiopia for most
of the sample, so the model takes it as given rather than trying to explain it.
"""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR

from emsp.processing.diagnostics import log_diff
from emsp.processing.regimes import make_event_dummies, make_seasonal_dummies

# Cholesky ordering: most exogenous first. Ethiopia cannot move world prices.
ENDOGENOUS = ["brent_usd_bbl", "urea_usd_mt", "cpi_index"]
EXCHANGE_RATE = "etb_per_usd"
FX_LAGS = 3  # current month + 3 previous months of depreciation
MIN_REGIME_MONTHS = 12  # don't estimate a regime effect from fewer months than this


def exchange_rate_terms(fx_change: pd.Series, max_lag: int = FX_LAGS) -> pd.DataFrame:
    """Current and lagged depreciation, one column per lag (lag0 = this month)."""
    return pd.DataFrame({f"etb_change_lag{k}": fx_change.shift(k) for k in range(max_lag + 1)})


def make_exog(fx_change: pd.Series) -> pd.DataFrame:
    """All external inputs for the dates in fx_change.

    Used for BOTH estimation and forecasting, so the two can never drift apart.
    """
    return pd.concat(
        [
            exchange_rate_terms(fx_change),
            make_event_dummies(fx_change.index)[["post_float_regime"]],
            make_seasonal_dummies(fx_change.index),
        ],
        axis=1,
    )


def prepare_var_data(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (endogenous monthly % changes, external inputs) on the same dates."""
    changes = log_diff(levels[ENDOGENOUS + [EXCHANGE_RATE]]).asfreq("MS")
    exog = make_exog(changes[EXCHANGE_RATE]).dropna()  # first FX_LAGS months lack lags

    # A regime effect estimated from a handful of months is unreliable, and the
    # forecast would apply it to every future month. Require enough data first.
    if exog["post_float_regime"].sum() < MIN_REGIME_MONTHS:
        exog = exog.drop(columns="post_float_regime")

    exog = exog.loc[:, exog.sum() != 0]  # drop any all-zero columns
    endog = changes.loc[exog.index, ENDOGENOUS]
    return endog, exog


def select_lag_order(changes: pd.DataFrame, exog: pd.DataFrame, max_lags: int = 12):
    """Compare lag lengths from 0 to max_lags using AIC, BIC, HQIC and FPE."""
    return VAR(changes, exog=exog).select_order(maxlags=max_lags)


def fit_var(changes: pd.DataFrame, exog: pd.DataFrame, lags: int):
    """Estimate the VARX with the given number of lags."""
    return VAR(changes, exog=exog).fit(lags)


def residual_checks(results, test_lags: int = 12) -> dict:
    """Key diagnostics: is the model stable, and do the residuals look like random noise?

    - stable: shocks fade out over time instead of exploding (must be True).
    - whiteness_pvalue: > 0.05 means no leftover pattern in the residuals (good).
    """
    whiteness = results.test_whiteness(nlags=test_lags, adjusted=True)
    return {
        "lags": results.k_ar,
        "stable": bool(results.is_stable()),
        "whiteness_pvalue": round(float(whiteness.pvalue), 4),
        "aic": round(float(results.aic), 3),
    }