"""Inflation forecasts conditional on an assumed path for the birr."""
from __future__ import annotations

import numpy as np
import pandas as pd

from emsp.modeling.var_model import ENDOGENOUS, EXCHANGE_RATE, make_exog
from emsp.processing.diagnostics import log_diff


def forecast_cpi(
    results,
    levels: pd.DataFrame,
    exog_columns,
    horizon: int = 12,
    monthly_depreciation_pct: float = 0.5,
) -> pd.DataFrame:
    """Forecast CPI for `horizon` months, assuming the birr weakens by a constant
    `monthly_depreciation_pct` each month.

    Returns the CPI index, monthly inflation and year-on-year inflation per month.
    """
    changes = log_diff(levels[ENDOGENOUS + [EXCHANGE_RATE]]).asfreq("MS")
    last_month = changes.index.max()
    future = pd.date_range(last_month + pd.DateOffset(months=1), periods=horizon, freq="MS")

    # Exchange-rate scenario: real history followed by the assumed depreciation path
    scenario = pd.Series(float(monthly_depreciation_pct), index=future)
    fx_path = pd.concat([changes[EXCHANGE_RATE], scenario])
    exog_future = make_exog(fx_path).loc[future, list(exog_columns)]  # same columns, same order

    recent = changes[ENDOGENOUS].to_numpy()[-results.k_ar:]
    predicted = pd.DataFrame(
        results.forecast(recent, steps=horizon, exog_future=exog_future.to_numpy()),
        index=future,
        columns=ENDOGENOUS,
    )

    # Monthly log-changes back to CPI levels, then year-on-year inflation
    cpi_history = levels["cpi_index"].loc[:last_month]
    cpi_path = cpi_history.iloc[-1] * np.exp(predicted["cpi_index"].cumsum() / 100)
    full_cpi = pd.concat([cpi_history, cpi_path])
    yoy = (full_cpi / full_cpi.shift(12) - 1) * 100

    return pd.DataFrame({
        "cpi_index": cpi_path,
        "monthly_inflation_pct": predicted["cpi_index"],
        "yoy_inflation_pct": yoy.loc[future],
    })