"""Tests for splicing and for the Yahoo fallback. No internet needed."""
import numpy as np
import pandas as pd
import pytest

from emsp.processing import dataset
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


def fake_sources(monkeypatch, yahoo):
    """Replace every download with small fake data, so the test needs no internet."""
    months = pd.date_range("2024-01-01", periods=14, freq="MS")
    official, _ = make_series()
    monkeypatch.setattr(dataset, "get_ethiopia_exchange_rate", lambda: official)
    monkeypatch.setattr(
        dataset, "get_ethiopia_cpi",
        lambda: pd.Series(np.arange(100.0, 114.0), index=months, name="cpi_index"),
    )
    monkeypatch.setattr(
        dataset, "get_commodity_prices",
        lambda refresh=False: pd.DataFrame(
            {"brent_usd_bbl": 80.0, "urea_usd_mt": 400.0}, index=months
        ),
    )
    monkeypatch.setattr(dataset, "get_yahoo_etb_monthly", yahoo)


def test_dataset_falls_back_to_imf_only_when_yahoo_fails(monkeypatch):
    def broken_yahoo():
        raise ConnectionError("Yahoo unreachable")

    fake_sources(monkeypatch, broken_yahoo)
    data, coverage = dataset.build_dataset()

    assert data.index.max() == pd.Timestamp("2024-12-01")  # stops where the IMF data stops
    assert "IMF only" in coverage.loc["etb_per_usd", "note"]


def test_dataset_uses_yahoo_when_it_validates(monkeypatch):
    _, extension = make_series()
    fake_sources(monkeypatch, lambda: extension)
    data, coverage = dataset.build_dataset()

    assert data.index.max() == pd.Timestamp("2025-02-01")  # two extra months from Yahoo
    assert "extended" in coverage.loc["etb_per_usd", "note"]