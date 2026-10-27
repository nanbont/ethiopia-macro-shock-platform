"""Tests for the impulse response helper, using simulated data."""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from emsp.modeling.irf import price_level_response


def fit_fake_model():
    rng = np.random.default_rng(0)
    index = pd.date_range("2001-01-01", periods=200, freq="MS")
    data = pd.DataFrame(rng.normal(size=(200, 2)), index=index, columns=["oil", "cpi"])
    return VAR(data).fit(1)


def test_shock_is_exactly_the_requested_size_on_impact():
    response = price_level_response(fit_fake_model(), "oil", "oil", horizon=12, shock_pct=10.0)
    assert abs(response.iloc[0] - 10.0) < 1e-9


def test_response_covers_every_month_of_the_horizon():
    response = price_level_response(fit_fake_model(), "oil", "cpi", horizon=12)
    assert len(response) == 13  # month 0 (impact) plus 12 months