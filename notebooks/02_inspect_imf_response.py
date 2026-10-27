"""Save the full IMF CPI response and decode its structure.

Goal: understand which time period each observation belongs to,
and whether the CPI values are really null.
"""
import json
from pathlib import Path

import requests

URL = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI/+/ETH.CPI._T.IX.M"

response = requests.get(URL, headers={"Accept": "application/json"}, timeout=30)
response.raise_for_status()  # stops with an error if the status is not 200
payload = response.json()

# 1. Save the raw response so we can study it without calling the API again
out_path = Path("data/raw/imf_cpi_eth_sample.json")
out_path.write_text(json.dumps(payload, indent=2))
print("Saved full response to", out_path)

data = payload["data"]
print("\nKeys inside 'data':", list(data.keys()))

# 2. Find the structure section (the "dictionary" that explains the numbers)
structure = data["structures"][0] if "structures" in data else data["structure"]
print("Keys inside structure:", list(structure.keys()))

time_dimension = structure["dimensions"]["observation"][0]
time_values = time_dimension["values"]
print("\nTime dimension:", time_dimension["id"], "| number of periods:", len(time_values))

measures = structure.get("measures", {}).get("observation", [])
attributes = structure.get("attributes", {}).get("observation", [])
print("Measures:  ", [m["id"] for m in measures])
print("Attributes:", [a["id"] for a in attributes])

# 3. Decode the first and last observations
series = data["dataSets"][0]["series"]
first_series = next(iter(series.values()))
observations = sorted(first_series["observations"].items(), key=lambda kv: int(kv[0]))
print("\nNumber of observations:", len(observations))

print("\nFirst 3 observations (period -> raw array):")
for index, raw in observations[:3]:
    print(" ", time_values[int(index)].get("id"), "->", raw)

print("\nLast 3 observations:")
for index, raw in observations[-3:]:
    print(" ", time_values[int(index)].get("id"), "->", raw)