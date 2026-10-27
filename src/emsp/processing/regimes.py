"""Documented exchange-rate policy events and seasonal effects, as dummy variables.

Only official policy events are included, to avoid data mining. Each event dummy
covers the announcement month and the following month, because a change made
partway through a month is split across two monthly averages.
"""
from __future__ import annotations

import pandas as pd

# Event name -> announcement month
POLICY_EVENTS = {
    "devaluation_2010": "2010-09-01",
    "devaluation_2017": "2017-10-01",
    "float_2024": "2024-07-01",  # float announced at the end of July 2024
}
EVENT_WINDOW_MONTHS = 2
FLOAT_START = pd.Timestamp("2024-08-01")  # first full month under the float


def make_event_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One 0/1 column per month of each policy event window, plus a post-float regime column.

    Each month gets its OWN dummy: a single dummy across two months would force the
    same adjustment on both, leaving equal-and-opposite residuals.
    """
    dummies = pd.DataFrame(index=index)
    for name, start in POLICY_EVENTS.items():
        window = pd.date_range(start, periods=EVENT_WINDOW_MONTHS, freq="MS")
        for number, month in enumerate(window, start=1):
            dummies[f"{name}_m{number}"] = (index == month).astype(float)

    dummies["post_float_regime"] = (index >= FLOAT_START).astype(float)
    return dummies


def make_seasonal_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One 0/1 column per calendar month, February to December.

    January is left out on purpose: the VAR already has a constant, and including
    all 12 months would duplicate it (the 'dummy variable trap').
    """
    return pd.DataFrame(
        {f"month_{m:02d}": (index.month == m).astype(float) for m in range(2, 13)},
        index=index,
    )