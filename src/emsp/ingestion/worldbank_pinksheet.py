"""Download and parse the World Bank Pink Sheet (monthly commodity prices).

The download URL changes every month, so we look it up from the commodity page.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

PAGE_URL = "https://www.worldbank.org/en/research/commodity-markets"
HEADERS = {"User-Agent": "Mozilla/5.0 (ethiopia-macro-shock-platform research project)"}
TIMEOUT_SECONDS = 60
PRICES_SHEET = "Monthly Prices"
DEFAULT_PATH = Path("data/raw/cmo_historical_monthly.xlsx")
MAX_AGE_DAYS = 7  # re-download if the saved file is older than this (the source updates monthly)

# World Bank column name -> our clean column name
SERIES = {
    "Crude oil, Brent": "brent_usd_bbl",
    "Urea": "urea_usd_mt",
}


def find_download_url(page_html: str, base_url: str = PAGE_URL) -> str:
    """Find the monthly Pink Sheet .xlsx link in the commodity page HTML."""
    links = re.findall(r'href="([^"]*CMO-Historical-Data-Monthly\.xlsx)"', page_html)
    if not links:
        raise ValueError("Could not find the Pink Sheet monthly download link on the page")
    return urljoin(base_url, links[0])


def download_pinksheet(out_path: Path = DEFAULT_PATH) -> Path:
    """Look up the current download link, then save the Excel file to disk."""
    page = requests.get(PAGE_URL, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    page.raise_for_status()
    url = find_download_url(page.text)

    file = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    file.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(file.content)
    return out_path


def parse_monthly_prices(raw: pd.DataFrame, series: dict[str, str] = SERIES) -> pd.DataFrame:
    """Turn the raw 'Monthly Prices' sheet (read with header=None) into a clean table.

    Finds the header row by name, skips the units row, parses '1960M01' dates,
    and converts placeholders like '...' to missing values.
    """
    # 1. Find the row that holds the commodity names
    first_name = next(iter(series))
    header_row = None
    for i in range(min(15, len(raw))):
        if first_name in set(raw.iloc[i].astype(str).str.strip()):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find a header row containing {first_name!r}")

    names = raw.iloc[header_row].astype(str).str.strip()  # strip fixes 'Urea '
    data = raw.iloc[header_row + 2 :]  # +2 skips the units row

    # 2. Parse dates like '1960M01'
    dates = pd.to_datetime(
        data.iloc[:, 0].astype(str).str.strip(), format="%YM%m", errors="coerce"
    )

    # 3. Pick each series by NAME, not by column number
    result = pd.DataFrame(index=dates)
    for wb_name, our_name in series.items():
        matches = names[names == wb_name].index
        if len(matches) != 1:
            raise ValueError(f"Expected one column named {wb_name!r}, found {len(matches)}")
        result[our_name] = pd.to_numeric(data[matches[0]], errors="coerce").to_numpy()

    result = result[result.index.notna()].sort_index()  # drop rows without a valid date
    result.index.name = "date"
    return result


def is_stale(path: Path, max_age_days: float = MAX_AGE_DAYS) -> bool:
    """True if the file was last downloaded more than `max_age_days` ago."""
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds > max_age_days * 24 * 60 * 60


def get_commodity_prices(path: Path = DEFAULT_PATH, refresh: bool = False) -> pd.DataFrame:
    """Monthly Brent oil and urea prices in nominal US dollars.

    Uses the saved file if it is recent; downloads a new copy if it is missing,
    older than MAX_AGE_DAYS, or refresh=True.
    """
    if refresh or not path.exists() or is_stale(path):
        download_pinksheet(path)
    raw = pd.read_excel(path, sheet_name=PRICES_SHEET, header=None)
    return parse_monthly_prices(raw)