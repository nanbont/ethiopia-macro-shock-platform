"""Tests for the VARX input construction."""
import pandas as pd

from emsp.modeling.var_model import exchange_rate_terms, prepare_var_data
from tests.test_forecast import make_fake_levels


def test_exchange_rate_terms_are_shifted_copies():
    fx = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4, freq="MS"))
    terms = exchange_rate_terms(fx, max_lag=2)

    assert list(terms.columns) == ["etb_change_lag0", "etb_change_lag1", "etb_change_lag2"]
    assert terms.iloc[3].tolist() == [4.0, 3.0, 2.0]  # April: this month, March, February
    assert terms["etb_change_lag2"].iloc[:2].isna().all()  # no data two months before Jan/Feb
    
def test_regime_dummy_needs_at_least_12_months_of_data():
    levels = make_fake_levels()
    _, short_exog = prepare_var_data(levels.loc[:"2025-03-01"])  # 8 post-float months
    _, full_exog = prepare_var_data(levels)  # 25 post-float months

    assert "post_float_regime" not in short_exog.columns
    assert "post_float_regime" in full_exog.columns