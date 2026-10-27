"""Statistical checks a VAR needs before we can trust it."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def log_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly percentage change, approximated as 100 * change in log."""
    return (np.log(df).diff() * 100).dropna()


def stationarity_table(df: pd.DataFrame, significance: float = 0.05) -> pd.DataFrame:
    """Run ADF and KPSS on every column and give a combined verdict."""
    rows = []
    for column in df.columns:
        series = df[column].dropna()

        adf_p = adfuller(series, autolag="AIC", result_object=False)[1]
        with warnings.catch_warnings():
            # KPSS warns when the p-value is outside its lookup table (it caps at 0.01 / 0.10)
            warnings.simplefilter("ignore")
            kpss_p = kpss(series, regression="c", nlags="auto")[1]

        adf_says_stationary = adf_p < significance
        kpss_says_stationary = kpss_p >= significance

        if adf_says_stationary and kpss_says_stationary:
            verdict = "stationary"
        elif not adf_says_stationary and not kpss_says_stationary:
            verdict = "non-stationary"
        else:
            verdict = "inconclusive"

        rows.append({"variable": column, "adf_p": round(adf_p, 3),
                     "kpss_p": round(kpss_p, 3), "verdict": verdict})
    return pd.DataFrame(rows).set_index("variable")