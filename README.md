# Waga · Ethiopia Macroeconomic Shock Platform

**How world prices reach Ethiopian households.**

*Waga* (ዋጋ) means *price* in Amharic. This project measures how shocks to world oil and
fertilizer prices, together with the birr exchange rate, pass through to consumer prices in
Ethiopia, and turns the results into an interactive dashboard for scenario analysis.

**Live dashboard:** [wagaethiopia.streamlit.app](https://wagaethiopia.streamlit.app/)

## Key findings

- **Oil:** a 10% rise in world oil prices raises Ethiopia's consumer price level by about
  **0.8%** within a year (90% bootstrap band: 0.32% to 1.25%). Most of the effect arrives
  within three months.
- **The September 2026 oil spike:** Brent rose from $90.9 to $116.8, an oil price surprise of
  **+23.6%** relative to the model's expectation. The model estimates this adds roughly
  **2 percentage points** to year-on-year inflation over the following year.
- **Fertilizer:** the estimated effect of urea prices is positive but **not statistically
  significant**, plausibly because much fertilizer reached farmers through government channels
  at administered prices.
- **Exchange rate:** pass-through from the birr to prices **fell after the July 2024 float**.
  Estimated over four months, it was about 0.23 before 2024 and about 0.07 in the full sample.
- **Forecasting, honestly:** in a rolling backtest from 2015, the model does **not** beat a
  naive "inflation stays where it is" benchmark as an unconditional forecaster (12-month RMSE
  10.5 vs 6.8 percentage points in normal periods), consistent with the inflation-forecasting
  literature. It is therefore presented as a tool for **scenario analysis**, where it relies on
  the tested shock-transmission estimates.

## The dashboard

| Page | What it shows |
|---|---|
| Overview | Latest inflation, birr rate and commodity prices, with data freshness by source |
| Shock transmission | Impulse responses of Ethiopian prices to oil and urea shocks, with 90% bands |
| Scenarios | Baseline vs scenario inflation paths for an oil surprise and a birr depreciation path |
| Methodology | Model specification, live diagnostics, backtest results, limitations and sources |

## Method

- **Model:** a VARX(1) on monthly log-changes of Brent crude oil, urea and Ethiopian CPI
  (Cholesky order: oil → urea → CPI, since Ethiopia cannot move world prices).
- **Exchange rate as an external input:** for most of the sample the National Bank of Ethiopia
  set the birr administratively, so the model takes the depreciation path (current month plus
  three lags) as given instead of trying to explain it.
- **Structural break:** a post-float regime dummy from August 2024, used only once 12 months of
  post-float data exist.
- **Seasonality:** monthly dummies, added after residual diagnostics showed a 12-month pattern
  in CPI.
- **Uncertainty:** residual bootstrap with 500 replications, resampling whole months.
- **Evaluation:** rolling-origin backtest with no look-ahead, reported separately for normal
  periods and for forecast windows spanning the 2024 float. Alternative lag lengths (3 and 6)
  and a 50/50 model–benchmark combination were chosen in advance and reported in full.

## Data

| Series | Source |
|---|---|
| Consumer price index | IMF SDMX 3.0 API (`CPI` dataflow), [data.imf.org](https://data.imf.org) |
| Birr per US dollar | IMF SDMX 3.0 API (`ER` dataflow); months after the IMF's latest release are filled from Yahoo Finance (`ETB=X`), accepted only if it matches the IMF within 2% over the previous 12 months |
| Brent crude oil, urea | World Bank Commodity Price Data (the Pink Sheet), [worldbank.org/en/research/commodity-markets](https://www.worldbank.org/en/research/commodity-markets) |

As a validation check, year-on-year inflation computed from the IMF series for December 2025
(9.7%) matches the figure reported in the IMF Executive Director's statement cited below.

If Yahoo Finance is unavailable, the pipeline falls back to official IMF data only and the
dashboard says so.

## Known limitations

- Only about two years of post-float data exist, so post-float pass-through cannot yet be
  estimated separately with precision.
- The joint residual whiteness test rejects, driven by a small lag-2 pattern in the urea
  equation; the CPI equation passes.
- A few fat-tailed months (e.g. 2009) remain without dummies, since no documented policy event
  explains them.
- The most recent Yahoo Finance exchange-rate values should be re-checked against official
  releases as they appear.

## Run it locally

```bash
git clone https://github.com/nanbont/waga.git
cd waga
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
streamlit run app/streamlit_app.py
```

## Project structure

```
src/emsp/
  ingestion/    IMF SDMX client, World Bank Pink Sheet client, Yahoo Finance extension
  processing/   alignment, stationarity diagnostics, policy-event and seasonal dummies
  modeling/     VARX model, impulse responses, bootstrap, forecasts, backtest
  viz/          shared chart styling
app/            Streamlit dashboard (display only; all calculations live in src/emsp)
notebooks/      exploration and the reproducible backtest evaluation
tests/          35 offline tests (no internet needed)
```

## Policy context

This project empirically examines the commodity and exchange-rate channels highlighted in:

Regis O. N'Sonde (Executive Director), Fatou Assah (Alternate Executive Director) and
Mawek Tesfaye Mengistu (Advisor to the Executive Director) (2026). *Statement by the Executive
Director on the Federal Democratic Republic of Ethiopia,* January 16, 2026. In *The Federal
Democratic Republic of Ethiopia: Fourth Review Under the Extended Credit Facility Arrangement*
(IMF Country Report No. 26/20). International Monetary Fund.
https://doi.org/10.5089/9798229035569.002

The source document is not redistributed in this repository; please use the DOI link.

## Author

**Nanbon Tadele Alemu**, Computer Science, University of Messina.
Interested in data science and quantitative research applied to macroeconomics and development,
especially in Africa.

[LinkedIn](https://www.linkedin.com/in/nanbon-tadele-a75844233/ · [GitHub](https://github.com/nanbont)