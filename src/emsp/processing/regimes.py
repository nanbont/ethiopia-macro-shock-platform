"""Documented exchange-rate policy events, encoded as dummy variables for the VAR.

Only official policy events are included, to avoid data mining.
"""
from __future__ import annotations

import pandas as pd

# Month of the largest monthly move for each documented policy event
POLICY_EVENTS = {
    "2010-09-01": "devaluation_2010",
    "2017-10-01": "devaluation_2017",
    "2024-08-01": "float_2024",
}
FLOAT_START = pd.Timestamp("2024-08-01")


def make_event_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return one 0/1 column per policy event, plus a post-float regime column."""
    columns = list(POLICY_EVENTS.values()) + ["post_float_regime"]
    dummies = pd.DataFrame(0.0, index=index, columns=columns)

    for month, name in POLICY_EVENTS.items():
        timestamp = pd.Timestamp(month)
        if timestamp in index:
            dummies.loc[timestamp, name] = 1.0

    dummies.loc[index >= FLOAT_START, "post_float_regime"] = 1.0
    return dummies