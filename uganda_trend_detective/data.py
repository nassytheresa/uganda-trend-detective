"""Download monthly NASA POWER data for a point and aggregate by season/year."""
import calendar
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Variable key -> (NASA POWER parameter, label, unit, how to aggregate a season)
VARIABLES = {
    "Rainfall": ("PRECTOTCORR", "Rainfall", "mm", "sum"),
    "Temperature": ("T2M", "Mean air temperature", "°C", "mean"),
}

# Season name -> list of months
SEASONS = {
    "Annual (Jan–Dec)": list(range(1, 13)),
    "First rainy season (Mar–May)": [3, 4, 5],
    "Second rainy season (Sep–Nov)": [9, 10, 11],
    "Dry season (Dec–Feb)": [12, 1, 2],
    "Dry season (Jun–Aug)": [6, 7, 8],
}

# Locations across Uganda (name -> (lat, lon)); spread out so each sits in its own data cell
REGIONS = {
    "Kampala": (0.3476, 32.5825),
    "Jinja": (0.4244, 33.2041),
    "Masaka": (-0.3136, 31.7350),
    "Mbarara": (-0.6072, 30.6545),
    "Kabale": (-1.2490, 29.9899),
    "Kasese": (0.1833, 30.0833),
    "Fort Portal": (0.6710, 30.2750),
    "Hoima": (1.4331, 31.3524),
    "Masindi": (1.6744, 31.7150),
    "Gulu": (2.7746, 32.2990),
    "Kitgum": (3.2889, 32.8870),
    "Arua": (3.0201, 30.9110),
    "Lira": (2.2499, 32.8998),
    "Soroti": (1.7147, 33.6111),
    "Mbale": (1.0821, 34.1751),
    "Moroto": (2.5345, 34.6666),
}


def fetch_monthly(lat, lon, parameter, start_year, end_year) -> pd.DataFrame:
    """Return DataFrame[year, month, value] of monthly means from NASA POWER."""
    params = {
        "parameters": parameter,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_year,
        "end": end_year,
        "format": "JSON",
    }
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(POWER_URL, params=params, timeout=60)
            r.raise_for_status()
            raw = r.json()["properties"]["parameter"][parameter]
            break
        except Exception as e:  # network hiccup or rate limit: wait and retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        raise last_err
    rows = []
    for key, val in raw.items():
        year, month = int(key[:4]), int(key[4:])
        if month == 13:  # POWER's annual-mean pseudo-month
            continue
        rows.append((year, month, np.nan if val == -999 else val))
    return pd.DataFrame(rows, columns=["year", "month", "value"])


def seasonal_series(monthly: pd.DataFrame, months, agg) -> pd.Series:
    """Aggregate monthly values into one value per year for the chosen months.

    Rainfall in POWER is mm/day, so 'sum' converts to mm per month first.
    A year is dropped if any month in the season is missing.
    """
    df = monthly[monthly["month"].isin(months)].copy()
    if 12 in months and 1 in months:
        # Season spans the new year: count December towards the following year
        df.loc[df["month"] == 12, "year"] += 1
    if agg == "sum":
        df["value"] = df.apply(
            lambda r: r["value"] * calendar.monthrange(int(r["year"]), int(r["month"]))[1],
            axis=1,
        )
    grouped = df.groupby("year")["value"]
    counts = grouped.count()
    full = counts[counts == len(months)].index
    out = grouped.sum() if agg == "sum" else grouped.mean()
    return out.loc[full]


def fetch_all(parameter, start_year, end_year, workers=4) -> dict:
    """Download monthly data for every location in REGIONS (a few at a time)."""
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fetch_monthly, lat, lon, parameter, start_year, end_year): name
            for name, (lat, lon) in REGIONS.items()
        }
        for f in as_completed(futures):
            try:
                out[futures[f]] = f.result()
            except Exception:
                pass  # missing locations are reported by the app
    return out


def parse_giovanni_csv(source):
    """Parse a Giovanni 'Time Series, Area-Averaged' CSV (e.g. GPM IMERG monthly precipitation).

    Returns (DataFrame[year, month, value in mm/day], detected_unit).
    IMERG monthly is delivered in mm/hr, so values are converted to mm/day to match
    seasonal_series(); a median above 1 is assumed to already be mm/day.
    """
    if hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
    else:
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
    lines = text.splitlines()
    fill, start = -9999.9, None
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("fill value"):
            try:
                fill = float(line.split(",")[-1])
            except ValueError:
                pass
        if low.startswith("time"):
            start = i + 1
            break
    if start is None:
        raise ValueError("This does not look like a Giovanni time-series CSV (no 'time' header row found).")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), names=["time", "value"], header=None, usecols=[0, 1])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time"])
    df.loc[(df["value"] == fill) | (df["value"] < -9000), "value"] = np.nan
    unit = "mm/day" if df["value"].median() > 1.0 else "mm/hr"
    if unit == "mm/hr":
        df["value"] = df["value"] * 24.0
    out = pd.DataFrame({"year": df["time"].dt.year, "month": df["time"].dt.month, "value": df["value"]})
    return out.reset_index(drop=True), unit
