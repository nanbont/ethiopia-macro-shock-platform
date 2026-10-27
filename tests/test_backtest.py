"""Tests for the backtest and scenario paths, using simulated data."""
import numpy as np
import pandas as pd

from emsp.modeling.backtest import run_backtest, summarize_backtest
from emsp.modeling.forecast import forecast_cpi
from emsp.modeling.var_model import fit_var, prepare_var_data
from tests.test_forecast import make_fake_levels


def test_constant_rate_and_constant_path_give_the_same_forecast():
    levels = make_fake_levels()
    endog, exog = prepare_var_data(levels)
    results = fit_var(endog, exog, lags=1)
    future = pd.date_range("2026-09-01", periods=12, freq="MS")

    as_number = forecast_cpi(results, levels, exog.columns, 12, 0.7)
    as_path = forecast_cpi(results, levels, exog.columns, 12, pd.Series(0.7, index=future))
    assert np.allclose(as_number.to_numpy(), as_path.to_numpy())


def test_backtest_has_one_row_per_origin_and_horizon():
    levels = make_fake_levels().loc[:"2016-12-01"]
    backtest = run_backtest(levels, start="2015-06-01", horizon=3)

    assert len(backtest) == 16 * 3  # origins Jun 2015 to Sep 2016, 3 horizons each
    assert (backtest["target"] > backtest["origin"]).all()  # never forecasts the past


def test_summary_has_one_row_per_horizon():
    levels = make_fake_levels().loc[:"2016-12-01"]
    summary = summarize_backtest(run_backtest(levels, start="2015-06-01", horizon=3))

    assert list(summary.index) == [1, 2, 3]
    assert (summary["model_rmse"] > 0).all()


def test_forecasts_crossing_the_float_are_flagged():
    levels = make_fake_levels()
    backtest = run_backtest(levels, start="2024-01-01", horizon=3)
    flag = backtest.set_index(["origin", "horizon"])["spans_float"]

    assert flag.loc[(pd.Timestamp("2024-04-01"), 3)]  # targets July 2024
    assert flag.loc[(pd.Timestamp("2024-07-01"), 1)]  # targets August 2024
    assert not flag.loc[(pd.Timestamp("2024-08-01"), 1)]  # float already in the data
    assert not flag.loc[(pd.Timestamp("2024-01-01"), 3)]  # ends before July 2024