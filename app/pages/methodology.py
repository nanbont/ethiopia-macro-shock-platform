"""Methodology: model, diagnostics, backtest, data sources and limitations."""
import pandas as pd
import streamlit as st
from statsmodels.stats.diagnostic import acorr_ljungbox

from emsp.modeling.var_model import residual_checks
from shared import load_backtest_summary, load_data, load_model

st.title("Methodology")

data, coverage, _ = load_data()
results, endog, exog = load_model(data)

# --- Model --------------------------------------------------------------------------------
st.header("Model")
st.markdown(
    f"""
A **VARX(1)** (vector autoregression with exogenous inputs), estimated on
**{len(endog)} months** ({endog.index[0]:%b %Y} to {endog.index[-1]:%b %Y}) of monthly log-changes:

- **Endogenous variables**, in Cholesky order: Brent crude oil → urea → Ethiopian CPI.
  Ethiopia is a small open economy: its inflation cannot move world prices within the month.
- **Exogenous inputs:** the birr/US\\$ depreciation rate (current month and three lags), a
  post-float regime dummy (from August 2024, used only once 12 months of data exist), and
  monthly seasonal dummies.
- **Why the birr is exogenous:** for most of the sample the National Bank of Ethiopia set the
  rate administratively. When modelled as an endogenous variable, its residuals showed long
  runs matching changes in the managed crawl, so the model takes the birr path as given.
- **Lag length:** 1, chosen by BIC and HQIC (AIC preferred more lags; backtest results with
  3 and 6 lags were similar).
- **Confidence bands:** residual bootstrap, 500 replications, resampling whole months.
"""
)

# --- Diagnostics --------------------------------------------------------------------------
st.header("Diagnostics")
checks = residual_checks(results)
col1, col2, col3 = st.columns(3)
col1.metric("Stable", "Yes" if checks["stable"] else "No")
col2.metric("Residual whiteness p-value", f"{checks['whiteness_pvalue']:.3f}")
col3.metric("AIC", f"{checks['aic']:.2f}")
ljung_box = pd.DataFrame({
    "Ljung–Box p-value (12 lags)": [
        float(acorr_ljungbox(results.resid[col], lags=[12], return_df=True)["lb_pvalue"].iloc[0])
        for col in results.resid.columns
    ]
}, index=["Brent equation", "Urea equation", "CPI equation"]).round(3)
st.dataframe(ljung_box, width="stretch")
st.caption(
    "A p-value above 0.05 means no evidence of leftover autocorrelation in that equation. "
    "The CPI equation is the one the analysis relies on; seasonal dummies were added after "
    "diagnostics showed a 12-month pattern in its residuals."
)

# --- Backtest -----------------------------------------------------------------------------
st.header("Backtest")
st.markdown(
    "Rolling-origin evaluation from January 2015: at each month the model is re-estimated on "
    "past data only and forecasts year-on-year inflation 1 to 12 months ahead, given the actual "
    "birr path. The benchmark assumes inflation stays at its latest value. RMSE is in "
    "percentage points; a ratio below 1 means the model beats the benchmark."
)
summary = load_backtest_summary(data)
columns = {"model_rmse": "Model RMSE", "naive_rmse": "Naive RMSE",
           "model_vs_naive": "Model ÷ naive", "forecasts": "Forecasts"}
col_normal, col_float = st.columns(2)
col_normal.subheader("Normal periods")
col_normal.dataframe(summary["normal"].loc[[1, 3, 6, 12]].rename(columns=columns).round(2), width="stretch")
col_float.subheader("Windows containing the 2024 float")
col_float.dataframe(summary["float"].loc[[1, 3, 6, 12]].rename(columns=columns).round(2), width="stretch")
st.markdown(
    """
**Reading the result honestly:** as an unconditional forecaster the model does **not** beat the
naive benchmark, consistent with the forecasting literature (e.g. Atkeson and Ohanian, 2001).
A pre-registered comparison of 1, 3 and 6 lags and a 50/50 model–naive combination did not
change this materially. The model is therefore presented as a tool for **scenario analysis**,
where it relies on the statistically tested shock-transmission estimates.

Forecasts made before the float performed worst: exchange-rate pass-through estimated before
2024 (about 0.23 over four months) was roughly three times larger than in the full sample
(about 0.07), so pre-float models over-predicted inflation after the 57% depreciation.
"""
)

# --- Limitations --------------------------------------------------------------------------
st.header("Known limitations")
st.markdown(
    """
- **Structural break:** only about two years of post-float data exist; post-float
  pass-through cannot yet be estimated separately with precision.
- **Urea effect** on prices is positive but not statistically significant at 90%.
- **Exchange-rate data after the IMF's latest release** come from Yahoo Finance, accepted only
  after matching the IMF within 2% over the previous 12 months. Yahoo's most recent values
  should be re-checked against official releases.
- **Fat-tailed months** (e.g. 2009) remain in the data without dummies, since no documented
  policy event explains them; the bootstrap does not assume normal errors.
"""
)

# --- Sources and citation -------------------------------------------------------------------
st.header("Data sources")
st.markdown(
    """
- **Consumer price index and birr/US\\$:** IMF SDMX 3.0 API, [data.imf.org](https://data.imf.org)
  (dataflows `CPI` and `ER`)
- **Brent crude oil and urea:** World Bank Commodity Price Data (the Pink Sheet),
  [worldbank.org/en/research/commodity-markets](https://www.worldbank.org/en/research/commodity-markets)
- **Recent birr/US\\$ extension:** Yahoo Finance (`ETB=X`), validated against the IMF series
"""
)

st.header("Policy context")
st.markdown(
    """
This project empirically examines the commodity and exchange-rate channels highlighted in:

Regis O. N'Sonde (Executive Director), Fatou Assah (Alternate Executive Director) and
Mawek Tesfaye Mengistu (Advisor to the Executive Director) (2026). *Statement by the Executive
Director on the Federal Democratic Republic of Ethiopia,* January 16, 2026. In *The Federal
Democratic Republic of Ethiopia: Fourth Review Under the Extended Credit Facility Arrangement*
(IMF Country Report No. 26/20). International Monetary Fund.
[doi.org/10.5089/9798229035569.002](https://doi.org/10.5089/9798229035569.002)
"""
)