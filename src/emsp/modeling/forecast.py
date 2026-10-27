"""Inflation forecasts conditional on an assumed birr path and an optional oil shock."""
from __future__ import annotations

import numpy as np
import pandas as pd

from emsp.modeling.var_model import ENDOGENOUS, EXCHANGE_RATE, make_exog
from emsp.processing.diagnostics import log_diff

OIL = "brent_usd_bbl"


def forecast_cpi(
    results,
    levels: pd.DataFrame,
    exog_columns,
    horizon: int = 12,
    monthly_depreciation_pct: float | pd.Series = 0.5,
    oil_shock_pct: float = 0.0,
) -> pd.DataFrame:
    """Forecast CPI for `horizon` months.

    - `monthly_depreciation_pct`: one number (same every month) or a Series with one
      value per forecast month (e.g. the actual path in a backtest).
    - `oil_shock_pct`: an oil price surprise in the first forecast month. Because the
      model is linear, the scenario equals the baseline plus the impulse response.
    """
    changes = log_diff(levels[ENDOGENOUS + [EXCHANGE_RATE]]).asfreq("MS")
    last_month = changes.index.max()
    future = pd.date_range(last_month + pd.DateOffset(months=1), periods=horizon, freq="MS")

    if isinstance(monthly_depreciation_pct, (int, float)):
        scenario = pd.Series(float(monthly_depreciation_pct), index=future)
    else:
        scenario = pd.Series(monthly_depreciation_pct).reindex(future)
        if scenario.isna().any():
            raise ValueError("The depreciation path must cover every forecast month")

    # Exchange-rate path: real history followed by the scenario
    fx_path = pd.concat([changes[EXCHANGE_RATE], scenario])
    exog_future = make_exog(fx_path).loc[future, list(exog_columns)]  # same columns, same order

    recent = changes[ENDOGENOUS].to_numpy()[-results.k_ar:]
    predicted = pd.DataFrame(
        results.forecast(recent, steps=horizon, exog_future=exog_future.to_numpy()),
        index=future,
        columns=ENDOGENOUS,
    )

    if oil_shock_pct:
        # Add the response of every variable to an oil shock of the requested size
        orth = results.irf(horizon - 1).orth_irfs  # (months, response, impulse)
        oil = list(results.names).index(OIL)
        scale = oil_shock_pct / orth[0, oil, oil]
        predicted = predicted + orth[:, :, oil] * scale

    # Monthly log-changes back to CPI levels, then year-on-year inflation
    cpi_history = levels["cpi_index"].loc[:last_month]
    cpi_path = cpi_history.iloc[-1] * np.exp(predicted["cpi_index"].cumsum() / 100)
    full_cpi = pd.concat([cpi_history, cpi_path])
    yoy = (full_cpi / full_cpi.shift(12) - 1) * 100

    return pd.DataFrame({
        "cpi_index": cpi_path,
        "monthly_inflation_pct": predicted["cpi_index"],
        "yoy_inflation_pct": yoy.loc[future],
        "oil_change_pct": predicted[OIL],
    })


def oil_surprise_pct(
    results,
    levels: pd.DataFrame,
    exog_columns,
    next_month_oil_price: float,
    monthly_depreciation_pct: float = 0.5,
) -> float:
    """How much an observed next-month oil price differs from what the model expected.

    Returns the surprise in % (log-change units), ready to pass as `oil_shock_pct`.
    """
    baseline = forecast_cpi(
        results, levels, exog_columns, horizon=1, monthly_depreciation_pct=monthly_depreciation_pct
    )
    actual_change = 100 * np.log(next_month_oil_price / levels[OIL].iloc[-1])
    return float(actual_change - baseline["oil_change_pct"].iloc[0])