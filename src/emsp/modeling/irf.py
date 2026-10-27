"""Impulse responses: how a commodity price shock passes through to Ethiopian prices."""
from __future__ import annotations

import pandas as pd


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