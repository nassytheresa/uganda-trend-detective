from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (REGIONS, SEASONS, VARIABLES, fetch_all, fetch_monthly,
                  parse_giovanni_csv, seasonal_series)
from trend import mann_kendall

st.set_page_config(page_title="Uganda Trend Detective", page_icon="🛰️", layout="wide")

DATA_START, DATA_END = 1990, 2025  # download once, then slice by the year slider


# ------------------------------------------------------------------ data helpers
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_monthly(region, parameter):
    lat, lon = REGIONS[region]
    return fetch_monthly(lat, lon, parameter, DATA_START, DATA_END)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_all(parameter):
    return fetch_all(parameter, DATA_START, DATA_END)


def series_for(monthly, season, agg, start, end):
    m = monthly[(monthly["year"] >= start) & (monthly["year"] <= end)]
    return seasonal_series(m, SEASONS[season], agg)


def analyze_all(var_key, season, start, end, alpha):
    param, _, _, agg = VARIABLES[var_key]
    all_monthly = load_all(param)
    rows = []
    for name, (lat, lon) in REGIONS.items():
        if name not in all_monthly:
            continue
        try:
            s = series_for(all_monthly[name], season, agg, start, end)
            r = mann_kendall(s.index, s.values, alpha)
            rows.append(
                dict(Location=name, lat=lat, lon=lon, slope=r.slope_per_year * 10,
                     p=r.p_value, sig=r.significant, n=r.n)
            )
        except ValueError:
            continue
    return pd.DataFrame(rows), all_monthly


# ------------------------------------------------------------------ text + charts
def verdict(var_key, season, region, res, unit, start, end):
    per_decade = res.slope_per_year * 10
    direction = "increased" if per_decade > 0 else "decreased"
    if "(" in season:
        what = f"{var_key} in {region}, {season.split(' (')[0].lower()} ({season.split('(')[-1].rstrip(')')})"
    else:
        what = f"{var_key} in {region}, {season.lower()}"
    if res.significant:
        return (
            f"✅ **Significant trend.** {what} has **{direction} by about "
            f"{abs(per_decade):.2f} {unit} per decade** between {start} and {end} "
            f"(p = {res.p_value:.3f}, below the {alpha_txt} threshold)."
        )
    return (
        f"⚪ **Not statistically significant.** {what} shows a slope of "
        f"{per_decade:+.2f} {unit} per decade, but p = {res.p_value:.3f} is above {alpha_txt}, so we "
        f"cannot rule out natural variability. That does not mean there is no change. "
        f"A longer time series can sometimes make a trend significant."
    )


def trend_chart(series, res, title, label, y_unit):
    x = series.index.values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=series.values, mode="lines+markers", name=label))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=res.intercept + res.slope_per_year * x,
            mode="lines",
            name="Sen's slope trend" + ("" if res.significant else " (not significant)"),
            line=dict(dash="solid" if res.significant else "dash", width=3),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=f"{label} ({y_unit})",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60),
    )
    return fig


def make_map(df, var_key, unit_short, selected):
    """Points coloured by trend; big = significant, faded = not significant."""
    lim = max(float(df["slope"].abs().max()), 1e-9)
    colorscale = "BrBG" if var_key == "Rainfall" else "RdBu_r"  # brown→green wetter / blue→red warmer
    new_api = hasattr(go, "Scattermap")
    Scatter = go.Scattermap if new_api else go.Scattermapbox

    def hover(d):
        return [
            f"<b>{r.Location}</b><br>Trend: {r.slope:+.1f} {unit_short}/decade<br>"
            f"p = {r.p:.3f} ({'significant' if r.sig else 'not significant'})"
            for r in d.itertuples()
        ]

    fig = go.Figure()
    sel = df[df["Location"] == selected]
    if len(sel):  # dark ring marks the location being inspected
        fig.add_trace(Scatter(lat=sel["lat"], lon=sel["lon"], mode="markers",
                              marker=dict(size=40, color="#111111", opacity=0.9),
                              hoverinfo="skip", showlegend=False))
    show_bar_on_sig = bool(df["sig"].any())
    for is_sig, size, opacity, name in [(False, 16, 0.55, "Not significant"),
                                        (True, 28, 0.95, "Significant")]:
        d = df[df["sig"] == is_sig]
        if d.empty:
            continue
        fig.add_trace(Scatter(
            lat=d["lat"], lon=d["lon"], mode="markers", name=name,
            marker=dict(size=size, opacity=opacity, color=d["slope"], colorscale=colorscale,
                        cmin=-lim, cmax=lim, showscale=(is_sig == show_bar_on_sig),
                        colorbar=dict(title=f"{unit_short}/decade", len=0.7)),
            customdata=d["Location"], hovertext=hover(d), hoverinfo="text",
        ))
    layout = dict(style="open-street-map", center=dict(lat=1.37, lon=32.3), zoom=5.7)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=620,
                      legend=dict(orientation="h", y=1.02, x=0))
    if new_api:
        fig.update_layout(map=layout)
    else:
        fig.update_layout(mapbox=layout)
    return fig


# ------------------------------------------------------------------ header + sidebar
st.title("🛰️ Uganda Trend Detective")
st.caption(
    "Is it changing? Where? By how much? Is the change significant? "
    "Built on NASA POWER (MERRA-2 reanalysis) data for the NASA Space Apps Challenge: "
    "*Be An Earth System Trend Detective!*"
)

with st.sidebar:
    st.header("1. Variable")
    var_key = st.selectbox("Variable", list(VARIABLES))
    st.header("2. Time period")
    season = st.selectbox("Season", list(SEASONS), index=2)
    start, end = st.slider("Years", 1990, 2025, (2001, 2025))
    alpha = st.selectbox("Significance level", [0.05, 0.01, 0.10], index=0)
    st.header("3. Region")
    region = st.selectbox("Location (for the explorer tab)", list(REGIONS))

alpha_txt = f"{alpha:g}"
param, label, unit, agg = VARIABLES[var_key]
y_unit = ("mm per year" if season.startswith("Annual") else "mm per season") if agg == "sum" else unit
unit_short = y_unit.split(" per")[0]

tab_map, tab_explore, tab_sat = st.tabs(["🗺️ Uganda map", "📈 Trend explorer", "🛰️ Satellite cross-check"])

# ------------------------------------------------------------------ TAB 1: MAP
with tab_map:
    st.markdown(f"**{label}, {season}, {start}–{end}.** Colour = size of the trend; "
                "large points = statistically significant; faded points = not significant. "
                "**Click a point** to inspect it.")
    try:
        with st.spinner("Fetching NASA data for all locations (first load takes ~30 s)…"):
            df, all_monthly = analyze_all(var_key, season, start, end, alpha)
    except Exception as e:
        st.error(f"Could not load NASA POWER data: {e}")
        df, all_monthly = pd.DataFrame(), {}

    if df.empty:
        st.warning("No locations could be analysed. Check your internet connection and try again.")
    else:
        missing_locs = [n for n in REGIONS if n not in set(df["Location"])]
        if missing_locs:
            st.caption("Not shown (data unavailable): " + ", ".join(missing_locs))

        up = int(((df["sig"]) & (df["slope"] > 0)).sum())
        down = int(((df["sig"]) & (df["slope"] < 0)).sum())
        flat = int((~df["sig"]).sum())
        word_up, word_down = ("wetter", "drier") if var_key == "Rainfall" else ("warmer", "cooler")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Significantly {word_up}", f"{up} of {len(df)}")
        m2.metric(f"Significantly {word_down}", f"{down} of {len(df)}")
        m3.metric("No significant trend", f"{flat} of {len(df)}")

        names = df["Location"].tolist()
        picked = st.session_state.get("picked", "Kampala" if "Kampala" in names else names[0])
        if picked not in names:
            picked = names[0]
        fig = make_map(df, var_key, unit_short, picked)
        event = st.plotly_chart(fig, key="uganda_map", on_select="rerun",
                                selection_mode="points", width="stretch")
        pts = []
        try:
            pts = event.selection.points
        except Exception:
            pass
        if pts:
            cd = pts[0].get("customdata")
            cd = cd[0] if isinstance(cd, (list, tuple)) else cd
            if cd in names and cd != picked:
                st.session_state["picked"] = cd
                st.rerun()

        choice = st.selectbox("Inspect location", names, index=names.index(picked))
        if choice != picked:
            st.session_state["picked"] = choice
            st.rerun()

        # ---- detail for the chosen point
        s = series_for(all_monthly[choice], season, agg, start, end)
        try:
            r = mann_kendall(s.index, s.values, alpha)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Trend, Sen's slope ({unit_short}/decade)", f"{r.slope_per_year * 10:+.1f}")
            c2.metric("p-value", f"{r.p_value:.3f}")
            c3.metric("Kendall's tau", f"{r.tau:+.2f}")
            c4.metric("Years of data", r.n)
            (st.success if r.significant else st.info)(
                verdict(var_key, season, choice, r, unit_short, int(s.index.min()), int(s.index.max())))
            st.plotly_chart(trend_chart(s, r, f"{label} – {choice}, {season}", label, y_unit),
                            width="stretch", key="map_detail")
        except ValueError as e:
            st.warning(str(e))

        with st.expander("All locations (table)"):
            table = df.rename(columns={"slope": f"Trend ({unit_short}/decade)", "p": "p-value",
                                       "sig": "Significant", "n": "Years"})
            table["Significant"] = table["Significant"].map({True: "Yes", False: "No"})
            table = table.drop(columns=["lat", "lon"]).round(3).sort_values("p-value")
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button("Download table (CSV)", table.to_csv(index=False).encode(),
                               file_name=f"uganda_{var_key}_{season}_{start}-{end}.csv".replace(" ", "_"))

# ------------------------------------------------------------------ TAB 3: SATELLITE CROSS-CHECK
DEFAULT_SAT = Path(__file__).parent / "data" / "kampala_gpm_imerg_monthly.csv"

with tab_sat:
    st.markdown(
        "**Do two independent NASA datasets agree?** The map uses NASA POWER (a model-based reanalysis). "
        "Here we compare it with **GPM IMERG**, satellite-measured rainfall calibrated with rain gauges, "
        "downloaded from NASA Giovanni for a box around Kampala (32.4–32.8°E, 0.2–0.5°N)."
    )
    if var_key != "Rainfall":
        st.info("The satellite cross-check is set up for rainfall. Switch **Variable** to Rainfall in the sidebar.")
    else:
        up = st.file_uploader("Use your own Giovanni CSV (optional)", type="csv")
        source = up if up is not None else (DEFAULT_SAT if DEFAULT_SAT.exists() else None)
        if source is None:
            st.info("Upload a Giovanni area-averaged time-series CSV to run the comparison.")
        else:
            try:
                sat_monthly, sat_unit = parse_giovanni_csv(source)
                pow_monthly = load_monthly("Kampala", param)
                s_pow = series_for(pow_monthly, season, agg, start, end)
                s_sat = series_for(sat_monthly, season, agg, start, end)
                common = s_pow.index.intersection(s_sat.index)
                if len(common) < 8:
                    st.warning("Fewer than 8 years overlap between the two datasets for this season and year range.")
                else:
                    a, b = s_pow.loc[common], s_sat.loc[common]
                    ra = mann_kendall(common, a.values, alpha)
                    rb = mann_kendall(common, b.values, alpha)
                    corr = float(np.corrcoef(a.values, b.values)[0, 1])

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Years compared", f"{len(common)} ({common.min()}–{common.max()})")
                    k2.metric("Year-to-year correlation (r)", f"{corr:+.2f}")
                    k3.metric("Satellite units detected", sat_unit)

                    cmp = pd.DataFrame({
                        "Dataset": ["NASA POWER (reanalysis)", "GPM IMERG (satellite + gauges)"],
                        f"Trend ({unit_short}/decade)": [round(ra.slope_per_year * 10, 1), round(rb.slope_per_year * 10, 1)],
                        "p-value": [round(ra.p_value, 3), round(rb.p_value, 3)],
                        "Significant": ["Yes" if ra.significant else "No", "Yes" if rb.significant else "No"],
                    })
                    st.dataframe(cmp, hide_index=True, width="stretch")

                    same_dir = np.sign(ra.slope_per_year) == np.sign(rb.slope_per_year)
                    if ra.significant and rb.significant and same_dir:
                        st.success("✅ **Confirmed.** Both independent datasets show a significant trend in the same direction.")
                    elif ra.significant != rb.significant:
                        who = "POWER" if ra.significant else "the satellite data"
                        st.warning(
                            f"⚠️ **Not confirmed.** Only {who} shows a significant trend. When two NASA datasets "
                            "disagree, the result is uncertain and should be reported with caution."
                        )
                    elif not ra.significant and not rb.significant:
                        st.info("⚪ **Neither dataset shows a significant trend** for this season and period.")
                    else:
                        st.warning("⚠️ **Conflicting results:** both are significant but in opposite directions.")

                    fig = go.Figure()
                    for name, ser, res_, color in [("NASA POWER", a, ra, "#1f77b4"),
                                                   ("GPM IMERG (satellite)", b, rb, "#e07b00")]:
                        fig.add_trace(go.Scatter(x=common, y=ser.values, mode="lines+markers", name=name,
                                                 line=dict(color=color)))
                        fig.add_trace(go.Scatter(x=common, y=res_.intercept + res_.slope_per_year * common.values,
                                                 mode="lines", name=f"{name} trend",
                                                 line=dict(color=color, dash="solid" if res_.significant else "dash", width=3)))
                    fig.update_layout(title=f"Kampala rainfall, {season}: model vs satellite",
                                      xaxis_title="Year", yaxis_title=f"Rainfall ({y_unit})",
                                      legend=dict(orientation="h", y=-0.2), margin=dict(t=60))
                    st.plotly_chart(fig, width="stretch", key="sat_chart")
                    st.caption(
                        "Differences are expected: POWER is a single ~50 km model grid cell, while the satellite "
                        "series averages a small box of 0.1° cells. Dashed trend line = not significant."
                    )
            except Exception as e:
                st.error(f"Could not run the cross-check: {e}")

with st.expander("Method and data"):
    st.markdown(
        """
- **Data:** NASA POWER monthly time series (rainfall: `PRECTOTCORR`, temperature: `T2M`), accessed via the POWER API, https://power.larc.nasa.gov.
- **Seasons:** monthly values are aggregated per year (rainfall summed to mm, temperature averaged). Years with missing months are dropped.
- **Trend test:** Mann-Kendall (two-sided, tie-corrected). **Trend size:** Sen's slope. A trend is *significant* when p is below the chosen level.
- **Map:** one point per location; colour = trend per decade, size = significance. Nearby points can share the same ~50 km data cell, so neighbouring results are not fully independent.
- **Caveats:** gridded reanalysis data (not rain-gauge observations); the test assumes roughly independent years, so we flag high autocorrelation; with many locations, a few "significant" results are expected by chance alone; significance depends on the length of the record.
"""
    )

# ------------------------------------------------------------------ TAB 2: EXPLORER
with tab_explore:
    try:
        with st.spinner("Fetching NASA data…"):
            series = series_for(load_monthly(region, param), season, agg, start, end)
    except Exception as e:
        st.error(f"Could not load NASA POWER data: {e}")
        st.stop()
    try:
        res = mann_kendall(series.index, series.values, alpha)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Trend, Sen's slope ({unit_short}/decade)", f"{res.slope_per_year * 10:+.1f}")
    c2.metric("p-value", f"{res.p_value:.3f}")
    c3.metric("Kendall's tau", f"{res.tau:+.2f}")
    c4.metric("Years of data", res.n)
    (st.success if res.significant else st.info)(
        verdict(var_key, season, region, res, unit_short, int(series.index.min()), int(series.index.max())))

    missing = sorted(set(range(start, end + 1)) - set(int(y) for y in series.index))
    if missing:
        st.caption("Years not used because the season has missing or incomplete NASA data: "
                   + ", ".join(str(y) for y in missing))

    st.plotly_chart(trend_chart(series, res, f"{label} – {region}, {season}", label, y_unit),
                    width="stretch", key="explorer_chart")
    if abs(res.lag1_autocorr) > 0.3:
        st.warning(
            f"Year-to-year values are correlated (lag-1 autocorrelation = {res.lag1_autocorr:.2f}). "
            "This can make the p-value look more significant than it is, so treat this result with caution.")
    st.download_button("Download this time series (CSV)", series.rename(label).to_csv().encode(),
                       file_name=f"{region}_{var_key}_{season}.csv".replace(" ", "_"))
