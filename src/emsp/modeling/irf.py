"""Impulse responses: how a commodity price shock passes through to Ethiopian prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def price_level_response(
    results, impulse: str, response: str, horizon: int = 24, shock_pct: float = 10.0
) -> pd.Series:
    """Cumulative % response of `response` to a `shock_pct`% shock in `impulse`.

    Uses the Cholesky (orthogonalized) impulse responses in the model's variable order,
    rescaled so the impulse variable itself moves by exactly `shock_pct` on impact.
    Cumulating monthly changes gives the effect on the price LEVEL.
    """
    names = list(results.names)
    i, j = names.index(impulse), names.index(response)
    orth = results.irf(horizon).orth_irfs  # shape: (horizon + 1, response, impulse)

    scale = shock_pct / orth[0, i, i]  # make the shock exactly shock_pct on impact
    cumulative = orth[:, j, i].cumsum() * scale

    return pd.Series(
        cumulative,
        index=pd.RangeIndex(horizon + 1, name="months_after_shock"),
        name=f"{response}_after_{impulse}",
    )


def simulate_from_residuals(
    results, endog: pd.DataFrame, exog: pd.DataFrame, residuals: np.ndarray
) -> pd.DataFrame:
    """Rebuild a dataset from the estimated model and a given sequence of residuals.

    With the original residuals in their original order, this reproduces the real data.
    """
    lags = results.k_ar
    deterministic_rows = ["const"] + list(exog.columns)
    det_coefs = results.params.loc[deterministic_rows].to_numpy()
    det_inputs = np.column_stack([np.ones(len(exog)), exog.to_numpy()])
    deterministic = det_inputs @ det_coefs  # constant + exchange rate + dummies, per month

    simulated = endog.to_numpy().copy()  # first `lags` rows stay as the real starting values
    for t in range(lags, len(endog)):
        value = deterministic[t] + residuals[t - lags]
        for lag in range(1, lags + 1):
            value = value + results.coefs[lag - 1] @ simulated[t - lag]
        simulated[t] = value
    return pd.DataFrame(simulated, index=endog.index, columns=endog.columns)


def bootstrap_price_level_response(
    results,
    endog: pd.DataFrame,
    exog: pd.DataFrame,
    impulse: str,
    response: str,
    horizon: int = 24,
    shock_pct: float = 10.0,
    replications: int = 500,
    coverage: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """Point estimate plus residual-bootstrap confidence band for price_level_response.

    Resamples whole months of residuals (keeping the equations together), rebuilds the
    data, re-estimates the model and recomputes the response `replications` times.
    """
    rng = np.random.default_rng(seed)  # fixed seed -> same bands every run (reproducible)
    residuals = results.resid.to_numpy()

    draws = []
    for _ in range(replications):
        months = rng.integers(0, len(residuals), size=len(residuals))
        fake = simulate_from_residuals(results, endog, exog, residuals[months])
        refit = VAR(fake, exog=exog).fit(results.k_ar)
        draws.append(price_level_response(refit, impulse, response, horizon, shock_pct).to_numpy())
    draws = np.array(draws)

    tail = (1 - coverage) / 2 * 100  # e.g. 5% in each tail for a 90% band
    estimate = price_level_response(results, impulse, response, horizon, shock_pct)
    return pd.DataFrame(
        {
            "lower": np.percentile(draws, tail, axis=0),
            "estimate": estimate.to_numpy(),
            "upper": np.percentile(draws, 100 - tail, axis=0),
        },
        index=estimate.index,
    )