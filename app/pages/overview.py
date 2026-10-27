"""Overview: where Ethiopian inflation, the birr and commodity prices stand today."""
import plotly.graph_objects as go
import streamlit as st

from emsp.viz.theme import ACCENT, HIGHLIGHT, MUTED, style_figure
from shared import load_data, yoy_inflation

st.title("Ethiopia Macroeconomic Shock Platform")
st.caption(
    "How global oil and fertilizer price shocks, and the birr exchange rate, "
    "pass through to consumer prices in Ethiopia."
)

data, coverage, prices = load_data()
inflation = yoy_inflation(data["cpi_index"]).dropna()

if coverage.loc["etb_per_usd", "note"].startswith("IMF only"):
    st.warning(
        "Recent exchange-rate data could not be loaded from Yahoo Finance, so the model uses "
        "official IMF data only, which ends earlier. See 'Data freshness' below."
    )

# --- Headline numbers ---------------------------------------------------------
latest = inflation.index[-1]
fx = data["etb_per_usd"]
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    f"Inflation, {latest:%b %Y}",
    f"{inflation.iloc[-1]:.1f}%",
    f"{inflation.iloc[-1] - inflation.iloc[-2]:+.1f} pp m/m",
    delta_color="inverse",
)
col2.metric(
    f"Birr per US$, {fx.index[-1]:%b %Y}",
    f"{fx.iloc[-1]:.1f}",
    f"{(fx.iloc[-1] / fx.iloc[-13] - 1) * 100:+.1f}% y/y",
    delta_color="inverse",
)
brent, urea = prices["brent_usd_bbl"], prices["urea_usd_mt"]
col3.metric(
    f"Brent oil, {brent.index[-1]:%b %Y}",
    f"${brent.iloc[-1]:.1f}",
    f"{(brent.iloc[-1] / brent.iloc[-2] - 1) * 100:+.1f}% m/m",
    delta_color="inverse",
)
col4.metric(
    f"Urea, {urea.index[-1]:%b %Y}",
    f"${urea.iloc[-1]:.0f}/t",
    f"{(urea.iloc[-1] / urea.iloc[-2] - 1) * 100:+.1f}% m/m",
    delta_color="inverse",
)

# --- Inflation history --------------------------------------------------------
fig = go.Figure(go.Scatter(x=inflation.index, y=inflation, name="Inflation", line=dict(color=ACCENT)))
fig.add_vline(x="2024-07-01", line_dash="dot", line_color=MUTED)
fig.add_annotation(x="2024-07-01", y=1, yref="paper", text="Birr floated (Jul 2024)",
                   showarrow=False, xanchor="right", font=dict(color=MUTED))
st.plotly_chart(style_figure(fig, "Year-on-year consumer price inflation", "%"), width="stretch")

# --- Commodity prices -----------------------------------------------------------
recent = prices.loc["2015-01-01":]
fig = go.Figure()
fig.add_trace(go.Scatter(x=recent.index, y=recent["brent_usd_bbl"], name="Brent ($/bbl)",
                         line=dict(color=ACCENT)))
fig.add_trace(go.Scatter(x=recent.index, y=recent["urea_usd_mt"], name="Urea ($/t, right axis)",
                         line=dict(color=HIGHLIGHT), yaxis="y2"))
fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
st.plotly_chart(style_figure(fig, "World commodity prices", "$ per barrel"), width="stretch")

# --- Data freshness -------------------------------------------------------------
st.subheader("Data freshness")
st.caption(
    "The model uses only months where every series is available. Birr/USD comes from the IMF; "
    "months after the IMF's latest release are filled with Yahoo Finance data, accepted only "
    "after it matches the IMF within 2% over the previous 12 months."
)
SOURCE_LABELS = {
    "cpi_index": "Consumer price index (IMF)",
    "etb_per_usd": "Birr per US$",
    "brent_usd_bbl": "Brent crude oil (World Bank)",
    "urea_usd_mt": "Urea (World Bank)",
}
freshness = coverage.rename(index=SOURCE_LABELS).assign(
    first=coverage["first"].dt.strftime("%b %Y").to_numpy(),
    last=coverage["last"].dt.strftime("%b %Y").to_numpy(),
)
st.dataframe(freshness, width="stretch")