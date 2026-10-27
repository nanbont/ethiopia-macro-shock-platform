"""Scratch exploration of the IMF SDMX 3.0 API.

Goal: find a URL that returns Ethiopia's monthly CPI and look at the raw response.
This is exploration code, not production code.
"""
import requests

BASE = "https://api.imf.org/external/sdmx/3.0"
SERIES_KEY = "ETH.CPI._T.IX.M"

# Two URL styles seen in the wild; we test both and keep whichever works.
candidate_urls = [
    f"{BASE}/data/dataflow/IMF.STA/CPI/+/{SERIES_KEY}",
    f"{BASE}/data/IMF.STA,CPI,+/{SERIES_KEY}",
]

headers = {"Accept": "application/json"}

for url in candidate_urls:
    print("\n" + "=" * 70)
    print("Trying:", url)
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as error:
        print("Network error:", error)
        continue

    print("Status code:  ", response.status_code)
    print("Content-Type: ", response.headers.get("Content-Type"))
    print("First 800 characters of the body:")
    print(response.text[:800])