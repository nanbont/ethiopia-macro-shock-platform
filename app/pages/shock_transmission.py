"""Shock transmission: how a world price shock reaches Ethiopian consumer prices."""
import plotly.graph_objects as go
import streamlit as st

from emsp.viz.theme import ACCENT, style_figure
from shared import load_data, load_irf_bands

st.title("Shock transmission")
st.markdown(
    "If a world commodity price jumps **today**, how much higher will Ethiopia's consumer "
    "price level be in the months that follow? The line is the model's estimate; the shaded "
    "area is a **90% confidence band** from 500 bootstrap replications."
)

data, _, _ = load_data()
bands = load_irf_bands(data)

shocks = {"Brent crude oil": "brent_usd_bbl", "Urea fertilizer": "urea_usd_mt"}
col_choice, col_size = st.columns(2)
choice = col_choice.radio("World price shock", list(shocks), horizontal=True)
size = col_size.slider("Size of the price increase (%)", min_value=5, max_value=50, value=10, step=5)

band = bands[shocks[choice]] * size / 10  # the model is linear, so responses scale with the shock

# --- Chart: estimate with confidence band ------------------------------------------
months = band.index
fig = go.Figure()
fig.add_trace(go.Scatter(x=months, y=band["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=months, y=band["lower"], line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(79, 195, 247, 0.2)", name="90% band"))
fig.add_trace(go.Scatter(x=months, y=band["estimate"], line=dict(color=ACCENT, width=3), name="Estimate"))
fig.add_hline(y=0, line_color="gray", line_width=1)
fig.update_xaxes(title="Months after the shock")
st.plotly_chart(
        style_figure(fig, f"Ethiopian price level after a {size}% {choice} price rise", "% change"),
    width="stretch",
)

# --- Key numbers -------------------------------------------------------------------------
cols = st.columns(3)
for col, month in zip(cols, [3, 6, 12]):
    row = band.loc[month]
    col.metric(f"After {month} months", f"{row['estimate']:+.2f}%",
               f"90% range {row['lower']:.2f} to {row['upper']:.2f}", delta_color="off")

if band.loc[12, "lower"] > 0:
    st.success("The whole 90% band is above zero: this effect is statistically significant.")
else:
    message = "The 90% band includes zero: the effect is plausible but not precisely estimated."
    if shocks[choice] == "urea_usd_mt":
        message += (
            " One likely reason is that much fertilizer reached farmers through government "
            "channels at administered prices, which blurs the link to world prices."
        )
    st.warning(message)

st.caption(
    "Model: VARX(1) on monthly log-changes of Brent, urea and CPI, with the birr exchange rate "
    "(current and three lags), a post-float regime dummy and seasonal dummies as external inputs. "
    "Shocks are identified by Cholesky ordering: world prices first, Ethiopian CPI last."
)