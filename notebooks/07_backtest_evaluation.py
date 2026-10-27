"""Backtest evaluation reported in the README.

1. Model vs naive benchmark, split into normal periods and windows spanning the 2024 float.
2. A pre-registered comparison of 1, 3 and 6 lags and a 50/50 model/naive combination.
   These three variants were chosen before seeing results, to avoid data snooping.
"""
import numpy as np
import pandas as pd

from emsp.modeling.backtest import run_backtest, summarize_backtest
from emsp.processing.dataset import build_dataset


def rmse(errors: pd.Series) -> float:
    return float(np.sqrt((errors**2).mean()))


data, _ = build_dataset()

# --- 1. Baseline model (1 lag) vs naive ---
backtest = run_backtest(data, start="2015-01-01", horizon=12, lags=1)
for label, subset in [
    ("ALL FORECASTS", backtest),
    ("NORMAL PERIODS", backtest[~backtest["spans_float"]]),
    ("WINDOWS CONTAINING THE FLOAT", backtest[backtest["spans_float"]]),
]:
    print(label)
    print(summarize_backtest(subset).loc[[1, 3, 6, 12]].round(2).to_string())
    print()

# --- 2. Pre-registered variants (normal periods only) ---
rows = []
for lags in [1, 3, 6]:
    variant = backtest if lags == 1 else run_backtest(data, start="2015-01-01", horizon=12, lags=lags)
    normal = variant[~variant["spans_float"]]
    for h in [1, 3, 6, 12]:
        d = normal[normal["horizon"] == h]
        combo = (d["model"] + d["naive"]) / 2
        rows.append({
            "lags": lags,
            "horizon": h,
            "model": rmse(d["model"] - d["actual"]),
            "combo_50_50": rmse(combo - d["actual"]),
            "naive": rmse(d["naive"] - d["actual"]),
        })
print("PRE-REGISTERED VARIANTS (normal periods, RMSE in percentage points)")
print(pd.DataFrame(rows).round(2).to_string(index=False))