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
GOLD, CYAN, INK, MUTED = "#f5c542", "#4cc9f0", "#e8ecf4", "#9fb0c9"

# drier / cooler = first colour, wetter / warmer = last colour
SCALES = {
    "Rainfall": [[0.0, "#e0a526"], [0.5, "#e9edf5"], [1.0, "#27c5b5"]],
    "Temperature": [[0.0, "#4cc9f0"], [0.5, "#e9edf5"], [1.0, "#ff6b35"]],
}

# ------------------------------------------------------------------ style
st.markdown(
    """
<style>
:root{--gold:#f5c542;--ink:#e8ecf4;--muted:#9fb0c9;--panel:#101a2e;--line:#26334f;}
.stApp{background:radial-gradient(1200px 600px at 15% -10%,#15233f 0%,#0a1020 55%,#070b16 100%);}
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
.block-container{padding-top:2.2rem;max-width:1250px;}
section[data-testid="stSidebar"]{background:#0b1424;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] h2{color:var(--gold)!important;font-size:.9rem;text-transform:uppercase;letter-spacing:.1em;}
.stApp p,.stApp li{color:var(--ink);}
[data-testid="stWidgetLabel"] p{color:var(--muted)!important;}
[data-testid="stMetric"]{background:linear-gradient(160deg,#121d34,#0e1729);border:1px solid var(--line);
  border-left:3px solid var(--gold);border-radius:12px;padding:.8rem 1rem;}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:.8rem;}
[data-testid="stMetricValue"]{color:#fff!important;}
button[data-baseweb="tab"]{font-size:1.02rem;}
button[data-baseweb="tab"] p{color:var(--muted)!important;}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--gold)!important;font-weight:600;}
div[data-baseweb="tab-highlight"]{background-color:var(--gold)!important;}
div[data-baseweb="select"]>div{background:var(--panel)!important;border-color:var(--line)!important;}
div[data-baseweb="select"] *{color:var(--ink)!important;}
div[data-baseweb="popover"] ul{background:#101a2e!important;}
li[role="option"]{color:var(--ink)!important;}
.hero{padding:1.4rem 1.7rem;border:1px solid var(--line);border-radius:16px;margin-bottom:1.1rem;
  background:linear-gradient(120deg,rgba(245,197,66,.12),rgba(76,201,240,.06) 60%,rgba(0,0,0,0));}
.hero .tag{color:var(--gold);font-size:.72rem;letter-spacing:.18em;font-weight:600;}
.hero h1{margin:.25rem 0 .3rem 0;font-size:2.5rem;color:#fff;padding:0;}
.hero h1 span{background:linear-gradient(90deg,#f5c542,#ffe9a8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stApp .hero p{color:var(--muted);margin:0 0 .8rem 0;font-size:1.02rem;}
.chip{display:inline-block;margin:0 .45rem .3rem 0;padding:.2rem .75rem;border:1px solid var(--line);
  border-radius:999px;color:var(--ink);font-size:.78rem;background:rgba(255,255,255,.04);}
.report{border:1px solid var(--line);border-left:3px solid var(--gold);border-radius:12px;
  padding:1rem 1.3rem;background:#0e1729;margin:.8rem 0;}
.report h4{margin:0 0 .6rem 0;color:var(--gold);}
.stApp .report p{color:var(--ink);line-height:1.55;margin:0 0 .6rem 0;}
.legend{color:var(--muted);font-size:.9rem;margin:.2rem 0 .6rem 0;}
</style>
""",
    unsafe_allow_html=True,
)


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


def bh_significant(p, alpha):
    """Benjamini-Hochberg false-discovery-rate control across many locations."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / m
    k = int(np.max(np.where(ok)[0])) + 1 if ok.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out


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
            rows.append(dict(Location=name, lat=lat, lon=lon, slope=r.slope_per_year * 10,
                             p=r.p_value, sig=r.significant, n=r.n))
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["fdr"] = bh_significant(df["p"].values, alpha)
    return df, all_monthly


# ------------------------------------------------------------------ text + charts
def season_phrase(season):
    return season.split(" (")[0].lower() + (f" ({season.split('(')[-1].rstrip(')')})" if "(" in season else "")


def verdict(var_key, season, region, res, unit, start, end):
    per_decade = res.slope_per_year * 10
    direction = "increased" if per_decade > 0 else "decreased"
    what = f"{var_key} in {region}, {season_phrase(season)}"
    if res.significant:
        return (f"✅ **Significant trend.** {what} has **{direction} by about {abs(per_decade):.2f} {unit} "
                f"per decade** between {start} and {end} (p = {res.p_value:.3f}, below the {alpha_txt} threshold).")
    return (f"⚪ **Not statistically significant.** {what} shows a slope of {per_decade:+.2f} {unit} per decade, "
            f"but p = {res.p_value:.3f} is above {alpha_txt}, so we cannot rule out natural variability. "
            f"That does not mean there is no change. A longer time series can sometimes make a trend significant.")


def style_fig(fig, title, ytitle):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,0.025)", title=title, xaxis_title="Year",
                      yaxis_title=ytitle, legend=dict(orientation="h", y=-0.22), margin=dict(t=60))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def trend_chart(series, res, title, label, y_unit):
    x = series.index.values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=series.values, mode="lines+markers", name=label,
                             line=dict(color=CYAN, width=2.5), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=x, y=res.intercept + res.slope_per_year * x, mode="lines",
                             name="Sen's slope trend" + ("" if res.significant else " (not significant)"),
                             line=dict(color=GOLD, width=3.5, dash="solid" if res.significant else "dash")))
    return style_fig(fig, title, f"{label} ({y_unit})")


def make_map(df, var_key, unit_short, selected):
    """Dark map: colour = trend, big glowing point = significant, faded = not significant."""
    lim = max(float(df["slope"].abs().max()), 1e-9)
    scale = SCALES[var_key]
    new_api = hasattr(go, "Scattermap")
    Scatter = go.Scattermap if new_api else go.Scattermapbox

    def hover(d):
        return [f"<b>{r.Location}</b><br>Trend: {r.slope:+.1f} {unit_short}/decade<br>p = {r.p:.3f} "
                f"({'significant' if r.sig else 'not significant'})" for r in d.itertuples()]

    def colored(d, size, opacity, **extra):
        return dict(size=size, opacity=opacity, color=d["slope"], colorscale=scale, cmin=-lim, cmax=lim, **extra)

    fig = go.Figure()
    sig = df[df["sig"]]
    if len(sig):  # soft glow under significant points
        fig.add_trace(Scatter(lat=sig["lat"], lon=sig["lon"], mode="markers", hoverinfo="skip", showlegend=False,
                              marker=colored(sig, 50, 0.22, showscale=False)))
    sel = df[df["Location"] == selected]
    if len(sel):  # gold ring on the inspected location
        fig.add_trace(Scatter(lat=sel["lat"], lon=sel["lon"], mode="markers", hoverinfo="skip", showlegend=False,
                              marker=dict(size=44, color=GOLD, opacity=1)))
        fig.add_trace(Scatter(lat=sel["lat"], lon=sel["lon"], mode="markers", hoverinfo="skip", showlegend=False,
                              marker=dict(size=36, color="#0a1020", opacity=1)))
    show_bar_on = bool(df["sig"].any())
    for is_sig, size, opacity, name in [(False, 15, 0.6, "Not significant"), (True, 27, 1.0, "Significant")]:
        d = df[df["sig"] == is_sig]
        if d.empty:
            continue
        fig.add_trace(Scatter(
            lat=d["lat"], lon=d["lon"], mode="markers+text", name=name,
            marker=colored(d, size, opacity, showscale=(is_sig == show_bar_on),
                           colorbar=dict(title=dict(text=f"{unit_short}/decade", font=dict(color=INK)),
                                         tickfont=dict(color=INK), len=0.7, thickness=14, x=0.99)),
            text=d["Location"], textposition="top right", textfont=dict(color=INK, size=12),
            customdata=d["Location"], hovertext=hover(d), hoverinfo="text",
            selected=dict(marker=dict(opacity=opacity)), unselected=dict(marker=dict(opacity=opacity)),
        ))
    layout = dict(style="carto-darkmatter", center=dict(lat=1.37, lon=32.3), zoom=5.7)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=640, paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.03, x=0.01, font=dict(color=INK),
                                  bgcolor="rgba(10,16,32,0.6)"))
    fig.update_layout(map=layout) if new_api else fig.update_layout(mapbox=layout)
    return fig


def detective_report(df, var_key, season, start, end, alpha, unit_short):
    n = len(df)
    up = df[df["sig"] & (df["slope"] > 0)].sort_values("slope", ascending=False)
    down = df[df["sig"] & (df["slope"] < 0)].sort_values("slope")
    w_up, w_dn = ("increase", "decrease")
    paras = [
        f"{var_key} in the {season_phrase(season)}, {start}–{end}: {len(up)} of {n} locations show a statistically "
        f"significant {w_up}, {len(down)} a significant {w_dn}, and {n - len(up) - len(down)} show no significant "
        f"trend (significance level {alpha:g})."
    ]
    if len(up):
        t = up.iloc[0]
        paras.append(f"The strongest {w_up} is at {t.Location}: {t.slope:+.1f} {unit_short} per decade (p = {t.p:.3f}).")
    if len(down):
        t = down.iloc[0]
        paras.append(f"The strongest {w_dn} is at {t.Location}: {t.slope:+.1f} {unit_short} per decade (p = {t.p:.3f}).")
    k = int(df["fdr"].sum())
    paras.append(f"Because {n} locations were tested at once, some significant results are expected by chance. After a "
                 f"Benjamini-Hochberg false-discovery-rate correction, {k} of {n} locations remain significant.")
    paras.append("Caveats: gridded NASA POWER reanalysis data over about 25 years; nearby locations can share a data cell. "
                 "Check the satellite cross-check tab before treating a trend as confirmed.")
    return paras


# ------------------------------------------------------------------ header + sidebar
st.markdown(
    """
<div class="hero">
<div class="tag">NASA SPACE APPS CHALLENGE 2026 · BE AN EARTH SYSTEM TREND DETECTIVE</div>
<h1>Uganda <span>Trend Detective</span></h1>
<p>Is it changing? Where? By how much? Is the change real, or just noise? Find out, with NASA data.</p>
<span class="chip">🛰️ NASA POWER (MERRA-2)</span><span class="chip">🌧️ GPM IMERG satellite check</span><span class="chip">📈 Mann-Kendall + Sen's slope</span><span class="chip">🧪 False-discovery correction</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1 · Variable")
    var_key = st.selectbox("Variable", list(VARIABLES))
    st.header("2 · Time period")
    season = st.selectbox("Season", list(SEASONS), index=2)
    start, end = st.slider("Years", 1990, 2025, (2001, 2025))
    alpha = st.selectbox("Significance level", [0.05, 0.01, 0.10], index=0)
    st.header("3 · Region")
    region = st.selectbox("Location (for the explorer tab)", list(REGIONS))

alpha_txt = f"{alpha:g}"
param, label, unit, agg = VARIABLES[var_key]
y_unit = ("mm per year" if season.startswith("Annual") else "mm per season") if agg == "sum" else unit
unit_short = y_unit.split(" per")[0]

tab_map, tab_explore, tab_sat = st.tabs(["🗺️ Uganda map", "📈 Trend explorer", "🛰️ Satellite cross-check"])

# ------------------------------------------------------------------ TAB 1: MAP
with tab_map:
    st.markdown(
        f'<div class="legend"><b>{label}, {season_phrase(season)}, {start}–{end}.</b> Colour = size of the trend · '
        "large glowing points = statistically significant · faded points = not significant · "
        "<b>click a point</b> to inspect it.</div>",
        unsafe_allow_html=True,
    )
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

        up = int((df["sig"] & (df["slope"] > 0)).sum())
        down = int((df["sig"] & (df["slope"] < 0)).sum())
        flat = int((~df["sig"]).sum())
        word_up, word_down = ("wetter", "drier") if var_key == "Rainfall" else ("warmer", "cooler")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Significantly {word_up}", f"{up} of {len(df)}")
        m2.metric(f"Significantly {word_down}", f"{down} of {len(df)}")
        m3.metric("No significant trend", f"{flat} of {len(df)}")
        m4.metric("Survive multiple-test correction", f"{int(df['fdr'].sum())} of {len(df)}")

        names = df["Location"].tolist()
        picked = st.session_state.get("picked", "Kampala" if "Kampala" in names else names[0])
        if picked not in names:
            picked = names[0]
        fig = make_map(df, var_key, unit_short, picked)
        event = st.plotly_chart(fig, key="uganda_map", on_select="rerun", selection_mode="points", width="stretch")
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
            st.plotly_chart(trend_chart(s, r, f"{label} – {choice}, {season_phrase(season)}", label, y_unit),
                            width="stretch", key="map_detail")
        except ValueError as e:
            st.warning(str(e))

        paras = detective_report(df, var_key, season, start, end, alpha, unit_short)
        st.markdown('<div class="report"><h4>📝 Detective report</h4>'
                    + "".join(f"<p>{p}</p>" for p in paras) + "</div>", unsafe_allow_html=True)
        st.download_button("Download report (.txt)", "\n\n".join(paras).encode(),
                           file_name=f"detective_report_{var_key}_{start}-{end}.txt".replace(" ", "_"))

        with st.expander("All locations (table)"):
            table = df.rename(columns={"slope": f"Trend ({unit_short}/decade)", "p": "p-value",
                                       "sig": "Significant", "fdr": "Significant after correction", "n": "Years"})
            for c in ["Significant", "Significant after correction"]:
                table[c] = table[c].map({True: "Yes", False: "No"})
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
        up_file = st.file_uploader("Use your own Giovanni CSV (optional)", type="csv")
        source = up_file if up_file is not None else (DEFAULT_SAT if DEFAULT_SAT.exists() else None)
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
                        st.warning(f"⚠️ **Not confirmed.** Only {who} shows a significant trend. When two NASA datasets "
                                   "disagree, the result is uncertain and should be reported with caution.")
                    elif not ra.significant and not rb.significant:
                        st.info("⚪ **Neither dataset shows a significant trend** for this season and period.")
                    else:
                        st.warning("⚠️ **Conflicting results:** both are significant but in opposite directions.")

                    fig = go.Figure()
                    for name, ser, res_, color in [("NASA POWER", a, ra, CYAN), ("GPM IMERG (satellite)", b, rb, GOLD)]:
                        fig.add_trace(go.Scatter(x=common, y=ser.values, mode="lines+markers", name=name,
                                                 line=dict(color=color, width=2.5), marker=dict(size=7)))
                        fig.add_trace(go.Scatter(x=common, y=res_.intercept + res_.slope_per_year * common.values,
                                                 mode="lines", name=f"{name} trend",
                                                 line=dict(color=color, width=3.5, dash="solid" if res_.significant else "dash")))
                    style_fig(fig, f"Kampala rainfall, {season_phrase(season)}: model vs satellite", f"Rainfall ({y_unit})")
                    st.plotly_chart(fig, width="stretch", key="sat_chart")
                    st.caption("Differences are expected: POWER is a single ~50 km model grid cell, while the satellite "
                               "series averages a small box of 0.1° cells. Dashed trend line = not significant.")
            except Exception as e:
                st.error(f"Could not run the cross-check: {e}")

# ------------------------------------------------------------------ METHOD
with st.expander("Method and data"):
    st.markdown(
        """
- **Data:** NASA POWER monthly time series (rainfall: `PRECTOTCORR`, temperature: `T2M`), accessed via the POWER API, https://power.larc.nasa.gov. Satellite check: GPM IMERG monthly precipitation (final run) via NASA Giovanni.
- **Seasons:** monthly values are aggregated per year (rainfall summed to mm, temperature averaged). Years with missing months are dropped.
- **Trend test:** Mann-Kendall (two-sided, tie-corrected). **Trend size:** Sen's slope. A trend is *significant* when p is below the chosen level.
- **Multiple testing:** the map tests 16 locations at once, so we also report how many remain significant after a Benjamini-Hochberg false-discovery-rate correction.
- **Caveats:** gridded reanalysis data (not rain-gauge observations); the test assumes roughly independent years, so we flag high autocorrelation; nearby locations can share a data cell; significance depends on the length of the record.
"""
    )

# ------------------------------------------------------------------ TAB 2: EXPLORER
with tab_explore:
    try:
        with st.spinner("Fetching NASA data…"):
            series = series_for(load_monthly(region, param), season, agg, start, end)
        res = mann_kendall(series.index, series.values, alpha)
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Could not load NASA POWER data: {e}")
    else:
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

        st.plotly_chart(trend_chart(series, res, f"{label} – {region}, {season_phrase(season)}", label, y_unit),
                        width="stretch", key="explorer_chart")
        if abs(res.lag1_autocorr) > 0.3:
            st.warning(f"Year-to-year values are correlated (lag-1 autocorrelation = {res.lag1_autocorr:.2f}). "
                       "This can make the p-value look more significant than it is, so treat this result with caution.")
        st.download_button("Download this time series (CSV)", series.rename(label).to_csv().encode(),
                           file_name=f"{region}_{var_key}_{season}.csv".replace(" ", "_"))
