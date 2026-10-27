"""Find the right IMF key for the Ethiopian birr / US dollar exchange rate.

Reuses our own IMF client. The ER codes changed in the May 2026 IMF update,
so we try a few candidate keys and see which one returns data.
"""
import requests

from emsp.ingestion.imf_sdmx import fetch_sdmx_json, parse_single_series

candidate_keys = [
    "ETH.XDC_USD.PA_RT.M",   # birr per USD, period average
    "ETH.USD_XDC.PA_RT.M",   # same idea, codes in the other order
    "ETH.XDC_USD.EOP_RT.M",  # birr per USD, end of period
]

for key in candidate_keys:
    print("\n" + "=" * 70)
    print("Trying ER key:", key)
    try:
        payload = fetch_sdmx_json("ER", key)
        series = parse_single_series(payload)
    except requests.HTTPError as error:
        print("HTTP error:", error.response.status_code)
        continue
    except (ValueError, KeyError, IndexError) as error:
        print("Parse problem:", type(error).__name__, error)
        continue

    print("Observations:", len(series))
    if series.empty:
        continue
    print("Range:", series.index.min().date(), "to", series.index.max().date())
    print("Around the July 2024 float:")
    print(series.loc["2024-05-01":"2024-09-01"])
    print("Latest 2 months:", series.tail(2).to_dict())