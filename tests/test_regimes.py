"""Tests for the policy-event and seasonal dummies."""
import pandas as pd

from emsp.processing.regimes import make_event_dummies, make_seasonal_dummies


def test_each_event_month_gets_its_own_dummy():
    index = pd.date_range("2024-06-01", "2024-10-01", freq="MS")
    dummies = make_event_dummies(index)

    assert list(dummies["float_2024_m1"]) == [0.0, 1.0, 0.0, 0.0, 0.0]  # July
    assert list(dummies["float_2024_m2"]) == [0.0, 0.0, 1.0, 0.0, 0.0]  # August
    assert list(dummies["post_float_regime"]) == [0.0, 0.0, 1.0, 1.0, 1.0]

def test_events_outside_the_index_are_ignored():
    index = pd.date_range("2020-01-01", "2020-03-01", freq="MS")
    assert make_event_dummies(index).to_numpy().sum() == 0.0


def test_seasonal_dummies_have_one_column_per_month_except_january():
    index = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    dummies = make_seasonal_dummies(index)

    assert dummies.shape == (12, 11)
    assert dummies.loc["2020-01-01"].sum() == 0.0  # January is the baseline
    assert (dummies.sum() == 1.0).all()  # each month appears exactly once