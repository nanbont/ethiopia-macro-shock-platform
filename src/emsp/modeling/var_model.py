"""VAR model: variable ordering, data preparation and lag selection."""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR

from emsp.processing.diagnostics import log_diff
from emsp.processing.regimes import make_event_dummies

# Cholesky ordering: most exogenous first. Ethiopia cannot move world prices.
VARIABLE_ORDER = ["brent_usd_bbl", "urea_usd_mt", "etb_per_usd", "cpi_index"]


def prepare_var_data(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn levels into monthly % changes and build the matching dummy variables."""
    changes = log_diff(levels[VARIABLE_ORDER]).asfreq("MS")  # tell statsmodels it's monthly
    exog = make_event_dummies(changes.index)
    exog = exog.loc[:, exog.sum() > 0]  # drop events outside the sample (all-zero columns)
    return changes, exog


def select_lag_order(changes: pd.DataFrame, exog: pd.DataFrame, max_lags: int = 12):
    """Compare lag lengths from 0 to max_lags using AIC, BIC, HQIC and FPE."""
    model = VAR(changes, exog=exog)
    return model.select_order(maxlags=max_lags)