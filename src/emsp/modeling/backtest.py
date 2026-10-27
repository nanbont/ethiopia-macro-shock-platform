"""Rolling-origin backtest: how accurate would our forecasts have been in the past?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emsp.modeling.forecast import forecast_cpi
from emsp.modeling.var_model import EXCHANGE_RATE, fit_var, prepare_var_data
from emsp.processing.diagnostics import log_diff


def yoy_inflation(cpi: pd.Series) -> pd.Series:
    """Year-on-year inflation in percent."""
    return (cpi / cpi.shift(12) - 1) * 100


def run_backtest(
    levels: pd.DataFrame, start: str = "2015-01-01", horizon: int = 12, lags: int = 1
) -> pd.DataFrame:
    """For each month from `start`, estimate on data up to that month only,
    forecast `horizon` months ahead, and record the forecast next to the outcome.

    The model receives the ACTUAL birr path (conditional forecast). The naive
    benchmark assumes inflation stays at its latest value.
    """
    actual_yoy = yoy_inflation(levels["cpi_index"])
    actual_fx = log_diff(levels[[EXCHANGE_RATE]])[EXCHANGE_RATE]
    origins = levels.loc[start:].index[:-horizon]  # only origins whose outcomes we know

    rows = []
    for origin in origins:
        history = levels.loc[:origin]  # the model sees nothing after the origin
        endog, exog = prepare_var_data(history)
        results = fit_var(endog, exog, lags)

        future = pd.date_range(origin + pd.DateOffset(months=1), periods=horizon, freq="MS")
        forecast = forecast_cpi(results, history, exog.columns, horizon, actual_fx.loc[future])

        for step, month in enumerate(future, start=1):
            rows.append({
                "origin": origin,
                "horizon": step,
                "target": month,
                "actual": actual_yoy.loc[month],
                "model": forecast.loc[month, "yoy_inflation_pct"],
                "naive": actual_yoy.loc[origin],
            })
    return pd.DataFrame(rows)


def summarize_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    """RMSE per forecast horizon for the model and the naive benchmark."""
    errors = backtest.assign(
        model_error=backtest["model"] - backtest["actual"],
        naive_error=backtest["naive"] - backtest["actual"],
    )
    summary = errors.groupby("horizon").agg(
        model_rmse=("model_error", lambda e: float(np.sqrt((e**2).mean()))),
        naive_rmse=("naive_error", lambda e: float(np.sqrt((e**2).mean()))),
        forecasts=("model_error", "size"),
    )
    summary["model_vs_naive"] = summary["model_rmse"] / summary["naive_rmse"]
    return summary