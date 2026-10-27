"""Client for the IMF SDMX 3.0 REST API (data.imf.org).

Fetches a single time series and returns it as a clean pandas Series.
"""
from __future__ import annotations

import pandas as pd
import requests

BASE_URL = "https://api.imf.org/external/sdmx/3.0"
DEFAULT_AGENCY = "IMF.STA"
TIMEOUT_SECONDS = 30


def fetch_sdmx_json(
    dataflow: str, key: str, agency: str = DEFAULT_AGENCY, version: str = "+"
) -> dict:
    """Download the raw SDMX-JSON response for one dataflow and series key."""
    url = f"{BASE_URL}/data/dataflow/{agency}/{dataflow}/{version}/{key}"
    response = requests.get(url, headers={"Accept": "application/json"}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def parse_period(period: str) -> pd.Timestamp:
    """Convert an IMF monthly period like '2026-M08' to 2026-08-01."""
    if "-M" not in period:
        raise ValueError(f"Expected a monthly period like '2026-M08', got {period!r}")
    year, month = period.split("-M")
    return pd.Timestamp(year=int(year), month=int(month), day=1)


def parse_single_series(payload: dict) -> pd.Series:
    """Turn an SDMX-JSON response containing ONE series into a pandas Series.

    Each observation array is [OBS_VALUE, PRECISION, DERIVATION_TYPE,
    REFERENCE_PERIOD, STATUS]; we keep only OBS_VALUE and skip missing values.
    """
    data = payload["data"]
    time_values = data["structures"][0]["dimensions"]["observation"][0]["values"]

    all_series = data["dataSets"][0]["series"]
    if len(all_series) != 1:
        raise ValueError(f"Expected exactly 1 series, got {len(all_series)}")
    observations = next(iter(all_series.values()))["observations"]

    records = {}
    for index, raw in observations.items():
        value = raw[0]
        if value is None:
            continue  # no data for this period
        period = time_values[int(index)]["value"]
        records[parse_period(period)] = float(value)  # values arrive as text

    series = pd.Series(records, dtype="float64").sort_index()
    series.index.name = "date"
    return series


def get_ethiopia_cpi() -> pd.Series:
    """Monthly headline CPI index for Ethiopia (Dec 2016 = 100)."""
    payload = fetch_sdmx_json("CPI", "ETH.CPI._T.IX.M")
    series = parse_single_series(payload)
    series.name = "cpi_index"
    return series 


def get_ethiopia_exchange_rate() -> pd.Series:
    """Monthly average exchange rate: Ethiopian birr per US dollar.

    Note: IMF publication of this series lags CPI; check the last date before modeling.
    """
    payload = fetch_sdmx_json("ER", "ETH.XDC_USD.PA_RT.M")
    series = parse_single_series(payload)
    series.name = "etb_per_usd"
    return series