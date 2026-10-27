"""VAR model: variable ordering, data preparation and lag selection."""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR

from emsp.processing.diagnostics import log_diff
from emsp.processing.regimes import make_event_dummies, make_seasonal_dummies

# Cholesky ordering: most exogenous first. Ethiopia cannot move world prices.
VARIABLE_ORDER = ["brent_usd_bbl", "urea_usd_mt", "etb_per_usd", "cpi_index"]


def prepare_var_data(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn levels into monthly % changes and build the matching dummy variables."""
    changes = log_diff(levels[VARIABLE_ORDER]).asfreq("MS")  # tell statsmodels it's monthly
    exog = pd.concat(
        [make_event_dummies(changes.index), make_seasonal_dummies(changes.index)], axis=1
    )
    exog = exog.loc[:, exog.sum() > 0]  # drop events outside the sample (all-zero columns)
    return changes, exog


def select_lag_order(changes: pd.DataFrame, exog: pd.DataFrame, max_lags: int = 12):
    """Compare lag lengths from 0 to max_lags using AIC, BIC, HQIC and FPE."""
    model = VAR(changes, exog=exog)
    return model.select_order(maxlags=max_lags)


def fit_var(changes: pd.DataFrame, exog: pd.DataFrame, lags: int):
    """Estimate a VAR with the given number of lags."""
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