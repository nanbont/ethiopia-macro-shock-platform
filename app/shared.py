"""Data and model loading shared by every dashboard page, with caching."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from emsp.ingestion.worldbank_pinksheet import get_commodity_prices
from emsp.modeling.irf import bootstrap_price_level_response
from emsp.modeling.var_model import fit_var, prepare_var_data
from emsp.processing.dataset import build_dataset

CACHE_SECONDS = 6 * 60 * 60  # refresh data at most every 6 hours


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Downloading data from the IMF and World Bank...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(aligned monthly dataset, coverage report, latest commodity prices)."""
    data, coverage = build_dataset()
    prices = get_commodity_prices()
    return data, coverage, prices


@st.cache_resource(show_spinner="Estimating the model...")
def load_model(data: pd.DataFrame):
    """Fit the baseline VARX (1 lag). Returns (results, endog, exog)."""
    endog, exog = prepare_var_data(data)
    return fit_var(endog, exog, lags=1), endog, exog


def yoy_inflation(cpi: pd.Series) -> pd.Series:
    """Year-on-year inflation in percent."""
    return (cpi / cpi.shift(12) - 1) * 100


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Computing confidence bands (bootstrap)...")
def load_irf_bands(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bootstrap bands for a 10% shock to oil and to urea. Responses scale linearly,
    so other shock sizes are obtained by multiplying."""
    results, endog, exog = load_model(data)
    return {
        shock: bootstrap_price_level_response(results, endog, exog, shock, "cpi_index", horizon=24)
        for shock in ["brent_usd_bbl", "urea_usd_mt"]
    }