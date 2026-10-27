"""Tests for the World Bank Pink Sheet parser. No internet needed."""
import os
import time

import pandas as pd
import pytest

from emsp.ingestion.worldbank_pinksheet import find_download_url, is_stale, parse_monthly_prices
def make_raw_sheet() -> pd.DataFrame:
    """A miniature copy of the real 'Monthly Prices' sheet, including its quirks."""
    rows = [
        ["World Bank Commodity Price Data", None, None, None],  # title row
        ["monthly prices in nominal US dollars", None, None, None],
        [None, "Crude oil, Brent", "DAP", "Urea "],  # names (note 'Urea ' with a space)
        [None, "($/bbl)", "($/mt)", "($/mt)"],  # units row
        ["2020M01", "64.0", "...", "250.0"],  # '...' placeholder, numbers as text
        ["2020M02", 55.5, 300.0, 245.0],
        [None, None, None, None],  # blank row at the bottom
    ]
    return pd.DataFrame(rows)


# --- find_download_url -------------------------------------------------------

def test_finds_absolute_download_link():
    html = '<a href="https://thedocs.worldbank.org/x/CMO-Historical-Data-Monthly.xlsx">Monthly</a>'
    assert find_download_url(html) == "https://thedocs.worldbank.org/x/CMO-Historical-Data-Monthly.xlsx"


def test_turns_relative_link_into_full_url():
    html = '<a href="/en/doc/x/CMO-Historical-Data-Monthly.xlsx">Monthly</a>'
    assert find_download_url(html) == "https://www.worldbank.org/en/doc/x/CMO-Historical-Data-Monthly.xlsx"


def test_raises_when_link_is_missing():
    with pytest.raises(ValueError):
        find_download_url("<html>no links here</html>")


# --- parse_monthly_prices ----------------------------------------------------

def test_parses_dates_values_and_trailing_space_in_names():
    result = parse_monthly_prices(make_raw_sheet())

    assert list(result.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert list(result["brent_usd_bbl"]) == [64.0, 55.5]
    assert list(result["urea_usd_mt"]) == [250.0, 245.0]  # found despite 'Urea '
    assert result["brent_usd_bbl"].dtype == "float64"


def test_converts_placeholder_dots_to_missing():
    result = parse_monthly_prices(make_raw_sheet(), series={"Crude oil, Brent": "brent", "DAP": "dap"})

    assert pd.isna(result["dap"].iloc[0])  # '...' became NaN
    assert result["dap"].iloc[1] == 300.0


def test_raises_when_a_requested_column_does_not_exist():
    with pytest.raises(ValueError):
        parse_monthly_prices(make_raw_sheet(), series={"Crude oil, Brent": "brent", "Gold": "gold"})
        
        
def test_a_new_file_is_not_stale(tmp_path):
    saved = tmp_path / "prices.xlsx"
    saved.write_bytes(b"placeholder")
    assert not is_stale(saved)


def test_a_ten_day_old_file_is_stale(tmp_path):
    saved = tmp_path / "prices.xlsx"
    saved.write_bytes(b"placeholder")
    ten_days_ago = time.time() - 10 * 24 * 60 * 60
    os.utime(saved, (ten_days_ago, ten_days_ago))  # pretend it was saved 10 days ago
    assert is_stale(saved)