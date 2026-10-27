"""Tests for the IMF SDMX parser. These use a fake payload, so no internet is needed."""
import pandas as pd
import pytest

from emsp.ingestion.imf_sdmx import parse_period, parse_single_series


def make_payload(observations: dict, periods: list[str]) -> dict:
    """Build a minimal fake SDMX-JSON response shaped like the real IMF one."""
    return {
        "data": {
            "structures": [
                {"dimensions": {"observation": [
                    {"id": "TIME_PERIOD", "values": [{"value": p} for p in periods]}
                ]}}
            ],
            "dataSets": [{"series": {"0:0:0:0:0": {"observations": observations}}}],
        }
    }


def test_parse_period_converts_monthly_string():
    assert parse_period("2026-M08") == pd.Timestamp("2026-08-01")


def test_parse_period_rejects_non_monthly_string():
    with pytest.raises(ValueError):
        parse_period("2026-Q1")


def test_parser_skips_missing_values_and_converts_text_to_numbers():
    payload = make_payload(
        observations={
            "0": [None, None, 0, "2016M12", None],     # missing -> should be skipped
            "1": ["100.0", None, 0, "2016M12", None],
            "2": ["101.5", None, 0, "2016M12", None],
        },
        periods=["2016-M11", "2016-M12", "2017-M01"],
    )
    series = parse_single_series(payload)

    assert list(series.index) == [pd.Timestamp("2016-12-01"), pd.Timestamp("2017-01-01")]
    assert list(series.values) == [100.0, 101.5]
    assert series.dtype == "float64"


def test_parser_rejects_response_with_multiple_series():
    payload = make_payload(observations={}, periods=[])
    payload["data"]["dataSets"][0]["series"]["1:0:0:0:0"] = {"observations": {}}

    with pytest.raises(ValueError):
        parse_single_series(payload)