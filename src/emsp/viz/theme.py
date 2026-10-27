"""Shared dark styling for every Plotly chart in the dashboard."""
from __future__ import annotations

import plotly.graph_objects as go

ACCENT = "#4FC3F7"  # main series
HIGHLIGHT = "#FFB74D"  # scenario / shock series
MUTED = "#90A4AE"  # benchmarks, bands, annotations


def style_figure(fig: go.Figure, title: str = "", y_title: str = "") -> go.Figure:
    """Apply the dashboard's dark theme and consistent layout to a figure."""
    fig.update_layout(
        template="plotly_dark",
        title=title,
        yaxis_title=y_title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=70),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    return fig