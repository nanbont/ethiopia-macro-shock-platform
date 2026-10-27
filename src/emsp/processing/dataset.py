"""Combine all ingested series into one aligned monthly dataset."""
from __future__ import annotations

import pandas as pd

from emsp.ingestion.imf_sdmx import get_ethiopia_cpi, get_ethiopia_exchange_rate
from emsp.ingestion.worldbank_pinksheet import get_commodity_prices


def coverage_report(frames: list[pd.Series | pd.DataFrame]) -> pd.DataFrame:
    """For each variable: first month, last month, and number of observations."""
    outer = pd.concat(frames, axis=1, join="outer")
    return pd.DataFrame({
        "first": outer.apply(lambda col: col.first_valid_index()),
        "last": outer.apply(lambda col: col.last_valid_index()),
        "observations": outer.count(),
    })


def assert_no_monthly_gaps(df: pd.DataFrame) -> None:
    """Raise an error if any month is missing between the first and last row."""
    expected = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing = expected.difference(df.index)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} missing months, e.g. {list(missing[:3])}")


def align_monthly(frames: list[pd.Series | pd.DataFrame]) -> pd.DataFrame:
    """Keep only months where every variable has a value (an 'inner join')."""
    combined = pd.concat(frames, axis=1, join="inner").dropna().sort_index()
    assert_no_monthly_gaps(combined)
    return combined


def build_dataset(refresh_prices: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all sources and return (aligned dataset, coverage report)."""
    frames = [
        get_ethiopia_cpi(),
        get_ethiopia_exchange_rate(),
        get_commodity_prices(refresh=refresh_prices),
    ]
    return align_monthly(frames), coverage_report(frames)