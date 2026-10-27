"""Tests for the policy-event dummies."""
import pandas as pd

from emsp.processing.regimes import make_event_dummies


def test_event_dummies_mark_the_right_months():
    index = pd.date_range("2024-06-01", "2024-10-01", freq="MS")
    dummies = make_event_dummies(index)

    assert dummies.loc["2024-08-01", "float_2024"] == 1.0
    assert dummies["float_2024"].sum() == 1.0  # exactly one month
    assert list(dummies["post_float_regime"]) == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_events_outside_the_index_are_ignored():
    index = pd.date_range("2020-01-01", "2020-03-01", freq="MS")
    dummies = make_event_dummies(index)

    assert dummies.to_numpy().sum() == 0.0  # no events in early 2020