"""Find, download, and inspect the World Bank Pink Sheet (monthly commodity prices).

The download URL changes every month, so we look it up from the commodity page.
"""
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

PAGE_URL = "https://www.worldbank.org/en/research/commodity-markets"
HEADERS = {"User-Agent": "Mozilla/5.0 (ethiopia-macro-shock-platform research project)"}

# 1. Find the current download link on the page
page = requests.get(PAGE_URL, headers=HEADERS, timeout=30)
print("Page status:", page.status_code)
links = sorted(set(re.findall(r'href="([^"]*CMO-Historical-Data-Monthly\.xlsx)"', page.text)))
print("Links found:", links)
if not links:
    raise SystemExit("No download link found on the page. Tell Claude!")

# 2. Download the Excel file into data/raw (gitignored)
xlsx_url = urljoin(PAGE_URL, links[0])
file = requests.get(xlsx_url, headers=HEADERS, timeout=60)
file.raise_for_status()
out_path = Path("data/raw/cmo_historical_monthly.xlsx")
out_path.write_bytes(file.content)
print(f"Saved {out_path} ({len(file.content) / 1e6:.1f} MB)")

# 3. Inspect the workbook
workbook = pd.ExcelFile(out_path)
print("\nSheets:", workbook.sheet_names)

for sheet in ["Monthly Prices", "Monthly Indices"]:
    if sheet not in workbook.sheet_names:
        print(f"\nSheet {sheet!r} not found")
        continue
    raw = pd.read_excel(out_path, sheet_name=sheet, header=None)
    print(f"\n=== {sheet} | shape {raw.shape} ===")
    print("Top-left corner:")
    print(raw.iloc[:8, :3])

    print("Columns mentioning oil or fertilizer:")
    for col in raw.columns:
        labels = " | ".join(str(v) for v in raw.iloc[:10, col] if pd.notna(v))
        if re.search(r"brent|crude oil, average|urea|dap|fertili", labels, re.IGNORECASE):
            print(f"  column {col}: {labels}")

    print("Last 3 rows (first 3 columns):")
    print(raw.iloc[-3:, :3])