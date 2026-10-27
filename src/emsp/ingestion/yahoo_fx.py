"""Recent birr/USD rates from Yahoo Finance, used only to extend the IMF series."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

TICKER = "ETB=X"


def get_yahoo_etb_monthly(start: str = "2023-01-01") -> pd.Series:
    """Monthly average birr per US dollar, complete months only."""
    daily = yf.download(TICKER, start=start, progress=False, auto_adjust=False)
    if daily.empty:
        raise ValueError(f"Yahoo Finance returned no data for {TICKER}")

    close = daily["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]

    monthly = close.resample("MS").mean().dropna()
    current_month = pd.Timestamp.today().normalize().replace(day=1)
    monthly = monthly[monthly.index < current_month]  # drop the unfinished current month

    monthly.index.name = "date"
    monthly.name = "etb_per_usd"
    return monthly