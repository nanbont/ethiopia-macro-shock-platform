"""Scenarios: inflation outlook under an oil price surprise and a birr depreciation path."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from emsp.modeling.forecast import forecast_cpi, oil_surprise_pct
from emsp.viz.theme import ACCENT, HIGHLIGHT, MUTED, style_figure
from shared import load_data, load_model, yoy_inflation

BASELINE_DEPRECIATION = 0.5  # % per month, roughly the birr's pace in early-mid 2026
HORIZON = 12

st.title("Scenarios")
st.markdown(
    "Compare a **baseline** inflation outlook with a **scenario** of your choice. "
    "The difference between the two lines comes from the model's estimated shock transmission."
)

data, _, prices = load_data()
results, _, exog = load_model(data)
last_month = data.index[-1]

# Default oil scenario: the observed surprise in the first month after the data ends, if known
next_month = last_month + pd.DateOffset(months=1)
observed_surprise = None
if next_month in prices.index:
    observed_surprise = oil_surprise_pct(
        results, data, exog.columns, prices.loc[next_month, "brent_usd_bbl"], BASELINE_DEPRECIATION
    )

OIL_MIN, OIL_MAX = -40, 60
default_oil = 0 if observed_surprise is None else int(round(observed_surprise))
default_oil = max(OIL_MIN, min(OIL_MAX, default_oil))  # keep the default inside the slider range

col_oil, col_fx = st.columns(2)
oil_shock = col_oil.slider(
    f"Oil price surprise in {next_month:%b %Y} (%)",
    min_value=OIL_MIN, max_value=OIL_MAX, step=1, value=default_oil,
)
if observed_surprise is not None:
       col_oil.caption(
        f"Default = the actual surprise: Brent moved from \\${data['brent_usd_bbl'].iloc[-1]:.1f} to "
        f"\\${prices.loc[next_month, 'brent_usd_bbl']:.1f}, which is {observed_surprise:+.1f}% more than "
        "the model expected."
    )
depreciation = col_fx.slider(
    "Birr depreciation (% per month)", min_value=0.0, max_value=3.0, step=0.25,
    value=BASELINE_DEPRECIATION,
)
col_fx.caption(f"Baseline assumes {BASELINE_DEPRECIATION}% per month and no oil surprise.")

baseline = forecast_cpi(results, data, exog.columns, HORIZON, BASELINE_DEPRECIATION)
scenario = forecast_cpi(results, data, exog.columns, HORIZON, depreciation, oil_shock_pct=float(oil_shock))

# --- Chart: recent history, baseline and scenario -------------------------------------
history = yoy_inflation(data["cpi_index"]).loc[last_month - pd.DateOffset(months=36):]
fig = go.Figure()
fig.add_trace(go.Scatter(x=history.index, y=history, name="Actual", line=dict(color=MUTED)))
for label, path, color, dash in [("Baseline", baseline, ACCENT, "dash"), ("Scenario", scenario, HIGHLIGHT, "solid")]:
    x = [last_month] + list(path.index)  # start each line from the last actual value
    y = [history.iloc[-1]] + list(path["yoy_inflation_pct"])
    fig.add_trace(go.Scatter(x=x, y=y, name=label, line=dict(color=color, dash=dash, width=3)))
st.plotly_chart(style_figure(fig, "Year-on-year inflation: baseline vs scenario", "%"), width="stretch")

# --- Key numbers -----------------------------------------------------------------------------
cols = st.columns(3)
for col, months in zip(cols, [3, 6, 12]):
    month = baseline.index[months - 1]
    base_value = baseline.loc[month, "yoy_inflation_pct"]
    scen_value = scenario.loc[month, "yoy_inflation_pct"]
    col.metric(f"Inflation in {month:%b %Y}", f"{scen_value:.1f}%",
               f"{scen_value - base_value:+.1f} pp vs baseline", delta_color="inverse")

st.info(
    "**How to read this.** In backtests, this model did not beat a simple 'inflation stays where it is' "
    "rule as an unconditional forecaster, which is common for inflation. Its strength is the "
    "**difference between scenarios**, which rests on the statistically tested shock-transmission "
    "estimates. Treat the lines as conditional projections, not predictions."
)