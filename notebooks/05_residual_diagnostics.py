"""Where is the leftover pattern in the VAR residuals?

Checks each equation separately: Ljung-Box test and autocorrelation at key lags.
"""
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from emsp.modeling.var_model import fit_var, prepare_var_data
from emsp.processing.dataset import build_dataset

data, _ = build_dataset()
changes, exog = prepare_var_data(data)
residuals = fit_var(changes, exog, lags=1).resid

rows = []
for column in residuals.columns:
    autocorr = acf(residuals[column], nlags=12)
    ljung_box_p = acorr_ljungbox(residuals[column], lags=[12], return_df=True)["lb_pvalue"].iloc[0]
    rows.append({
        "equation": column,
        "ljung_box_p": round(ljung_box_p, 4),
        "acf_lag1": round(autocorr[1], 2),
        "acf_lag2": round(autocorr[2], 2),
        "acf_lag6": round(autocorr[6], 2),
        "acf_lag12": round(autocorr[12], 2),
    })

print(pd.DataFrame(rows).set_index("equation"))
band = 1.96 / len(residuals) ** 0.5
print(f"\nAutocorrelations beyond +/- {band:.2f} are statistically significant.")


# --- Which months produce the largest exchange-rate surprises? ---
fx_residuals = residuals["etb_per_usd"]
print("\n8 largest exchange-rate residuals (by size):")
largest = fx_residuals.abs().nlargest(8).index
print(fx_residuals.loc[largest].round(2))
print(f"\nTypical residual size (median absolute): {fx_residuals.abs().median():.2f}")