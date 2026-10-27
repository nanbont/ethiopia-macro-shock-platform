"""Combine all ingested series into one aligned monthly dataset."""
from __future__ import annotations

import pandas as pd

from emsp.ingestion.imf_sdmx import get_ethiopia_cpi, get_ethiopia_exchange_rate
from emsp.ingestion.worldbank_pinksheet import get_commodity_prices
from emsp.ingestion.yahoo_fx import get_yahoo_etb_monthly


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


def extend_series(
    official: pd.Series,
    extension: pd.Series,
    check_months: int = 12,
    tolerance_pct: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Keep official data where it exists; fill later months from a second source.

    Before trusting the second source, compare both over the last `check_months`
    overlapping months and refuse if they differ by more than `tolerance_pct`.
    Returns (combined series, source label per month).
    """
    overlap = official.index.intersection(extension.index)[-check_months:]
    if len(overlap) < check_months:
        raise ValueError(f"Need {check_months} overlapping months to validate, found {len(overlap)}")

    gap_pct = ((extension.loc[overlap] / official.loc[overlap]) - 1).abs() * 100
    if gap_pct.max() > tolerance_pct:
        raise ValueError(f"Sources disagree by up to {gap_pct.max():.2f}% (limit {tolerance_pct}%)")

    later = extension[extension.index > official.index.max()]
    combined = pd.concat([official, later]).sort_index()
    combined.name = official.name
    source = pd.Series(
        ["official"] * len(official) + ["extension"] * len(later),
        index=combined.index,
        name="source",
    )
    return combined, source


def build_dataset(
    refresh_prices: bool = False, extend_fx: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all sources and return (aligned dataset, coverage report).

    extend_fx=True fills the IMF exchange rate after its last month with validated
    Yahoo Finance data. If Yahoo is unreachable or fails validation, the dataset
    falls back to official IMF data only, and the coverage report says so in its
    `note` column. Set extend_fx=False for an official-data-only robustness check.
    """
    fx = get_ethiopia_exchange_rate()
    fx_note = "IMF"
    if extend_fx:
        try:
            fx, _ = extend_series(fx, get_yahoo_etb_monthly())
            fx_note = "IMF, extended with validated Yahoo Finance data"
        except Exception as error:  # network failure, changed Yahoo format, or failed validation
            fx_note = f"IMF only (Yahoo Finance extension unavailable: {type(error).__name__})"

    frames = [get_ethiopia_cpi(), fx, get_commodity_prices(refresh=refresh_prices)]
    coverage = coverage_report(frames)
    coverage["note"] = ""
    coverage.loc[fx.name, "note"] = fx_note
    return align_monthly(frames), coverage