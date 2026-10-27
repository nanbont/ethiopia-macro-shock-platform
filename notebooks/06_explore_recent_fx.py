"""Can Yahoo Finance extend the IMF birr/USD series past September 2025?

We only trust it if it matches the IMF closely in the months where both exist.
"""
import pandas as pd
import yfinance as yf

from emsp.ingestion.imf_sdmx import get_ethiopia_exchange_rate

daily = yf.download("ETB=X", start="2023-01-01", progress=False, auto_adjust=False)
print("Daily rows downloaded:", len(daily))

close = daily["Close"]
if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
    close = close.iloc[:, 0]
yahoo_monthly = close.resample("MS").mean()  # monthly average, same as the IMF "PA" series

imf = get_ethiopia_exchange_rate()
compare = pd.DataFrame({"imf": imf, "yahoo": yahoo_monthly}).loc["2024-01-01":]
compare["diff_pct"] = (compare["yahoo"] / compare["imf"] - 1) * 100
print(compare.round(2).to_string())