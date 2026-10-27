"""Tests for splicing an official series with a second source. No internet needed."""
import numpy as np
import pandas as pd
import pytest

from emsp.processing.dataset import extend_series


def make_series():
    index = pd.date_range("2024-01-01", periods=14, freq="MS")
    official = pd.Series(np.arange(100.0, 112.0), index=index[:12], name="etb_per_usd")
    extension = pd.Series(np.arange(100.0, 114.0) * 1.01, index=index)  # 1% off, 2 extra months
    return official, extension


def test_official_data_is_kept_and_only_later_months_are_filled():
    official, extension = make_series()
    combined, source = extend_series(official, extension)

    assert np.allclose(combined.iloc[:12].to_numpy(), official.to_numpy())
    assert list(source) == ["official"] * 12 + ["extension"] * 2
    assert combined.name == "etb_per_usd"


def test_extension_is_rejected_when_sources_disagree():
    official, extension = make_series()
    with pytest.raises(ValueError):
        extend_series(official, extension * 1.10)  # 11% off: too far


def test_extension_is_rejected_without_enough_overlap():
    official, extension = make_series()
    with pytest.raises(ValueError):
        extend_series(official.iloc[:5], extension)