# Uganda Trend Detective
NASA Space Apps Challenge 2026 – *Be An Earth System Trend Detective!*

Pick a **variable**, **time period** and **region**. The app finds the trend, tests whether it is
**statistically significant** (Mann-Kendall + Sen's slope), and shows *where* in Uganda it is changing
on an interactive map.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Features
- **Uganda map:** 16 locations coloured by trend per decade; large = significant, faded = not. Click a point to inspect it.
- **Trend explorer:** time series, trend line, p-value and plain-language verdict for one location.
- **Satellite cross-check:** compares NASA POWER with GPM IMERG satellite rainfall (Giovanni CSV; Kampala example included in `data/`, or upload your own).
- Honest reporting: non-significant results are labelled as such; autocorrelation warning; CSV downloads.

## Data
NASA POWER monthly point data (PRECTOTCORR rainfall, T2M temperature): https://power.larc.nasa.gov

## Files
- `trend.py` – Mann-Kendall test (tie-corrected) and Sen's slope (validated against `pymannkendall`)
- `data.py`  – NASA POWER download (parallel, with retries) + seasonal aggregation
- `app.py`   – Streamlit app
- `data/kampala_gpm_imerg_monthly.csv` – GPM IMERG monthly precipitation (NASA Giovanni), Kampala box 32.4–32.8E, 0.2–0.5N, 2001–2025
