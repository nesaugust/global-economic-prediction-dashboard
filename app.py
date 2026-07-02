import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib

st.set_page_config(
    page_title="GEIP — Global Economic Intelligence Platform",
    page_icon="🌍",
    layout="wide",
)

pio.templates.default = "plotly_dark"

# ===============================
# GLOBAL STYLE — matches GEIP mockup (dark navy, glass cards, blue accent)
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 90% 0%, rgba(37,99,235,0.16), transparent 32%),
        radial-gradient(circle at 10% 100%, rgba(14,165,233,0.08), transparent 30%),
        linear-gradient(135deg, #030712 0%, #07111f 42%, #0b1220 100%);
    color: #F8FAFC;
}

.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1400px; }

/* ---------- Sidebar (left nav rail, like GEIP) ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030712 0%, #07111f 55%, #020617 100%);
    border-right: 1px solid rgba(148,163,184,0.14);
}
section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
section[data-testid="stSidebar"] h1 { font-size: 22px !important; font-weight: 800 !important; }

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(37,99,235,0.14); }

/* Hide Streamlit's native radio circle marker — it's always the first child
   of the label, regardless of the exact data-baseweb attribute Streamlit
   ships with, so target it structurally rather than by attribute. */
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Highlight the active nav row (progressive enhancement — harmless if
   :has() isn't supported, it just won't get the extra highlight). */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(37,99,235,0.20);
    border: 1px solid rgba(96,165,250,0.35);
}

.nav-disabled {
    color: #475569 !important;
    font-size: 14px;
    padding: 0.5rem 0.6rem;
    border-radius: 12px;
    cursor: not-allowed;
}

/* ---------- Top bar ---------- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 10px;
    margin-bottom: 6px;
}
.topbar .stSelectbox, .topbar .stTextInput { min-width: 160px; }

/* Text */
h1 { font-size: 40px !important; font-weight: 800 !important; letter-spacing: -1.1px; color: #F8FAFC !important; }
h2, h3 { color: #F8FAFC !important; font-weight: 750 !important; }
p, span, label { color: #CBD5E1 !important; }

.page-title { font-size: 40px; font-weight: 800; letter-spacing: -1.1px; color: #FFFFFF; margin-bottom: 2px; }
.page-subtitle { font-size: 15px; color: #93C5FD; margin-bottom: 18px; }

/* KPI cards with accent + sparkline slot */
.metric-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.92), rgba(15,23,42,0.92));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 20px;
    padding: 18px 20px 10px 20px;
    box-shadow: 0 18px 42px rgba(0,0,0,0.26);
}
.metric-card:hover { border-color: rgba(96,165,250,0.38); }
.metric-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.metric-icon {
    width: 34px; height: 34px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.metric-title { font-size: 12.5px; color: #93C5FD; font-weight: 700; letter-spacing: 0.02em; }
.metric-value { font-size: 27px; font-weight: 800; color: #FFFFFF; }
.metric-delta { font-size: 12.5px; font-weight: 700; margin-left: 8px; }
.metric-note { font-size: 11.5px; color: #64748B; margin-top: 2px; }
.delta-up { color: #34D399; }
.delta-down { color: #F87171; }

/* Section / panel wrapper */
.panel-title { font-size: 17px; font-weight: 800; color: #F8FAFC; margin-bottom: 2px; }
.panel-sub { font-size: 12px; color: #64748B; margin-bottom: 10px; }
.chart-caption { font-size: 14px; font-weight: 700; color: #F8FAFC; margin: 2px 0 6px 4px; }

/* Risk / status chips */
.risk-chip {
    border-radius: 18px; padding: 14px 12px; text-align: left;
    border: 1px solid rgba(148,163,184,0.16);
}
.risk-chip .rc-count { font-size: 24px; font-weight: 800; color: #FFFFFF; margin-top: 6px; }
.risk-chip .rc-label { font-size: 11.5px; color: #94A3B8; }

/* Inputs */
div[data-baseweb="select"] > div, input {
    background-color: #0F172A !important;
    border: 1px solid rgba(148,163,184,0.26) !important;
    border-radius: 12px !important;
    color: #F8FAFC !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white !important; border: none; border-radius: 12px;
    padding: 0.62rem 1.1rem; font-weight: 800;
    box-shadow: 0 12px 28px rgba(37,99,235,0.24);
}
.stButton > button:hover { background: linear-gradient(135deg, #1D4ED8, #1E40AF); }

/* Segmented range control (1Y/5Y/10Y/All) */
[data-testid="stHorizontalBlock"] div[role="radiogroup"] {
    flex-direction: row; gap: 4px;
}
div[role="radiogroup"] label {
    border: 1px solid rgba(148,163,184,0.2);
    border-radius: 8px !important;
    padding: 2px 10px !important;
    font-size: 12px !important;
}

/* Plotly card wrap */
.js-plotly-plot {
    background: linear-gradient(145deg, rgba(15,23,42,0.94), rgba(17,24,39,0.90)) !important;
    border-radius: 20px; overflow: hidden;
    border: 1px solid rgba(148,163,184,0.14);
    box-shadow: 0 18px 45px rgba(0,0,0,0.24);
}

/* Dataframes */
div[data-testid="stDataFrame"] { background: #0F172A !important; border-radius: 16px; overflow: hidden; border: 1px solid rgba(148,163,184,0.16); }
div[data-testid="stDataFrame"] thead tr th { background: #1E293B !important; color: #F8FAFC !important; font-weight: 800 !important; }
div[data-testid="stDataFrame"] tbody tr { background: #0F172A !important; }
div[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: #111827 !important; }
div[data-testid="stDataFrame"] td { color: #CBD5E1 !important; }
div[data-testid="stDataFrame"] tbody tr:hover { background: #1F2937 !important; }

header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ===============================
# DATA LOADING
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("processed_global_economy_data.csv")
    risk_df = pd.read_csv("Prediction_Data/risk_score_data.csv")
    score_df = pd.read_csv("Prediction_Data/economic_score_data.csv")
    cluster_df = pd.read_csv("Prediction_Data/country_cluster_data.csv")
    anomaly_df = pd.read_csv("Prediction_Data/anomaly_detection_data.csv")
    recession_df = pd.read_csv("Prediction_Data/recession_probability_data.csv")
    return df, risk_df, score_df, cluster_df, anomaly_df, recession_df


df, risk_df, score_df, cluster_df, anomaly_df, recession_df = load_data()

# Filter datasets by selected year
def get_year_filtered_data(year):
    risk_year_df = (
        risk_df[risk_df["Year"] == year]
        if "Year" in risk_df.columns
        else risk_df
    )

    cluster_year_df = (
        cluster_df[cluster_df["Year"] == year]
        if "Year" in cluster_df.columns
        else cluster_df
    )

    anomaly_year_df = (
        anomaly_df[anomaly_df["Year"] == year]
        if "Year" in anomaly_df.columns
        else anomaly_df
    )

    recession_year_df = (
        recession_df[recession_df["Year"] == year]
        if "Year" in recession_df.columns
        else recession_df
    )

    return risk_year_df, cluster_year_df, anomaly_year_df, recession_year_df

RISK_LEVEL_ALIASES = {
    "low": {"low", "low risk", "minimal", "safe"},
    "medium": {"medium", "moderate", "mid", "medium risk", "average"},
    "high": {"high", "high risk", "elevated"},
    "crisis": {"crisis", "critical", "very high", "severe", "crisis risk", "extreme"},
}


def _normalized_risk_series():
    return risk_df["Risk_Level"].astype(str).str.strip().str.lower()


def risk_level_counts():
    """Case/whitespace/synonym-tolerant counts for the four risk chips.
    Falls back to 0 for a bucket only if truly nothing in Risk_Level matches
    any known alias for it — which usually means the source CSV uses a label
    we haven't seen yet (surfaced via risk_level_raw_values() below)."""
    normalized = _normalized_risk_series()
    return {
        bucket: int(normalized.isin(aliases).sum())
        for bucket, aliases in RISK_LEVEL_ALIASES.items()
    }


def risk_level_raw_values():
    return sorted(risk_df["Risk_Level"].dropna().astype(str).unique().tolist())


if "Crisis_Alert" not in risk_df.columns:
    _normalized = _normalized_risk_series()
    risk_df["Crisis_Alert"] = np.where(
        _normalized.isin(RISK_LEVEL_ALIASES["crisis"] | RISK_LEVEL_ALIASES["high"]),
        "Crisis Warning",
        "Normal",
    )

# ---- Region mapping (used for the Inflation Heatmap, grouped like the mockup) ----
REGION_MAP = {
    # North America
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    # South America
    "Brazil": "South America", "Argentina": "South America", "Chile": "South America",
    "Colombia": "South America", "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Bolivia": "South America", "Paraguay": "South America", "Uruguay": "South America",
    # Europe
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe", "Italy": "Europe",
    "Spain": "Europe", "Netherlands": "Europe", "Switzerland": "Europe", "Sweden": "Europe",
    "Poland": "Europe", "Belgium": "Europe", "Austria": "Europe", "Norway": "Europe",
    "Denmark": "Europe", "Finland": "Europe", "Ireland": "Europe", "Portugal": "Europe",
    "Greece": "Europe", "Czech Republic": "Europe", "Romania": "Europe", "Hungary": "Europe",
    "Russia": "Europe", "Ukraine": "Europe",
    # Asia
    "China": "Asia", "India": "Asia", "Japan": "Asia", "South Korea": "Asia", "Indonesia": "Asia",
    "Saudi Arabia": "Asia", "Turkey": "Asia", "Thailand": "Asia", "Vietnam": "Asia",
    "Malaysia": "Asia", "Philippines": "Asia", "Pakistan": "Asia", "Bangladesh": "Asia",
    "Singapore": "Asia", "United Arab Emirates": "Asia", "Israel": "Asia", "Iran": "Asia",
    "Iraq": "Asia", "Qatar": "Asia", "Kazakhstan": "Asia",
    # Africa
    "South Africa": "Africa", "Nigeria": "Africa", "Egypt": "Africa", "Kenya": "Africa",
    "Morocco": "Africa", "Algeria": "Africa", "Ethiopia": "Africa", "Ghana": "Africa", "Angola": "Africa",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
}


def get_region(country):
    return REGION_MAP.get(country, "Other")


if "Region" not in df.columns:
    df["Region"] = df["Country"].map(get_region)

years_sorted = sorted(df["Year"].dropna().unique())
latest_year = int(df["Year"].max())
earliest_year = int(df["Year"].min())
latest_df = df[df["Year"] == latest_year].copy()
country_list = sorted(df["Country"].dropna().unique())


# ===============================
# HELPERS
# ===============================
def format_number(value, prefix="", pct=False):
    if pd.isna(value):
        return "N/A"
    if pct:
        return f"{value:.2f}%"
    if abs(value) >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{prefix}{value / 1_000:.2f}K"
    return f"{prefix}{value:.2f}"


def previous_year(year):
    earlier = [y for y in years_sorted if y < year]
    return max(earlier) if earlier else None


def scoped_frame(year):
    """Rows for a given year, scoped to Global (all countries) or a single Country
    depending on the top-bar view selector. This is what keeps every KPI/chart
    connected to the Year + View controls."""
    frame = df[df["Year"] == year]
    if st.session_state.get("view_mode") == "Country View":
        frame = frame[frame["Country"] == st.session_state.get("focus_country", selected_country)]
    return frame


def sparkline(series_df, x_col, y_col, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_df[x_col], y=series_df[y_col],
        mode="lines", line=dict(color=color, width=2.4),
        fill="tozeroy", fillcolor=color.replace(")", ",0.14)").replace("rgb", "rgba"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=54, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def kpi_card(icon, icon_bg, title, value, delta_pct, note, spark_df=None, spark_col=None, spark_color="rgb(59,130,246)"):
    up = delta_pct is not None and delta_pct >= 0
    delta_html = ""
    if delta_pct is not None:
        arrow = "&uarr;" if up else "&darr;"
        cls = "delta-up" if up else "delta-down"
        delta_html = f"<span class='metric-delta {cls}'>{arrow} {abs(delta_pct):.2f}%</span>"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-head">
                <div class="metric-icon" style="background:{icon_bg};">{icon}</div>
                <div class="metric-title">{title}</div>
            </div>
            <div><span class="metric-value">{value}</span>{delta_html}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if spark_df is not None and spark_col is not None and len(spark_df) > 1:
        st.plotly_chart(sparkline(spark_df, "Year", spark_col, spark_color), use_container_width=True, config={"displayModeBar": False})


def hero(title, subtitle):
    st.markdown(f"<div class='page-title'>{title}</div><div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def insight(text):
    st.markdown(
        f"<div style='background:linear-gradient(135deg, rgba(15,23,42,0.92), rgba(15,23,42,0.72));"
        f"border:1px solid rgba(96,165,250,0.24); border-left:4px solid #3B82F6; padding:14px 18px;"
        f"border-radius:16px; margin:10px 0 20px 0; color:#DBEAFE; line-height:1.55;'>{text}</div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height=430, title=None):
    """Apply GEIP dashboard styling and prevent Plotly from showing `undefined` titles.

    Use Streamlit/HTML headings above charts instead of Plotly's built-in title.
    This removes the small `undefined` text that can appear inside charts.
    """

    # Only print a custom caption if you intentionally pass a clean title.
    resolved_title = None
    if title not in [None, "", "undefined", "Undefined"]:
        resolved_title = title

    # Remove any existing Plotly Express title/annotation completely.
    fig.update_layout(title=dict(text=""))
    fig.layout.title.text = ""
    fig.layout.annotations = ()

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.18)",
        font=dict(family="Inter, Arial", size=13, color="#CBD5E1"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(color="#CBD5E1"),
        ),
        margin=dict(l=42, r=28, t=34, b=46),
        hovermode="x unified",
        colorway=["#3B82F6", "#38BDF8", "#818CF8", "#22C55E", "#F59E0B", "#EF4444"],
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="rgba(148,163,184,0.18)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#94A3B8"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
        showline=False,
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#94A3B8"),
    )

    try:
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    except Exception:
        pass

    if resolved_title:
        st.markdown(f"<div class='chart-caption'>{resolved_title}</div>", unsafe_allow_html=True)

    return fig


def styled_dataframe(dataframe, formats=None):
    if formats:
        st.dataframe(dataframe.style.format(formats), use_container_width=True)
    else:
        st.dataframe(dataframe, use_container_width=True)


def range_years(range_label, up_to_year):
    if range_label == "1Y":
        span = 1
    elif range_label == "5Y":
        span = 5
    elif range_label == "10Y":
        span = 10
    else:
        return earliest_year, up_to_year
    return max(earliest_year, up_to_year - span), up_to_year


# ===============================
# SIDEBAR NAV — icon rail like GEIP
# ===============================
st.sidebar.markdown("### 🌐  GEIP")

NAV_OPTIONS = [
    "🏠 Overview",
    "📊 Economy",
    "💹 Markets",
    "🌍 Trade",
    "🛡️ Risk Intelligence",
    "🤖 AI Predictions",
    "🧭 Country Analysis",
    "🗂️ Data Explorer",
]
nav_choice = st.sidebar.radio("Navigation", NAV_OPTIONS, label_visibility="collapsed")

st.sidebar.markdown(
    "<div class='nav-disabled'>📄 Reports</div>"
    "<div class='nav-disabled'>🔔 Alerts</div>"
    "<div class='nav-disabled'>⚙️ Settings</div>"
    "<div style='font-size:10.5px;color:#334155;padding:4px 6px;'>Coming soon</div>",
    unsafe_allow_html=True,
)

# ===============================
# TOP BAR — search (decorative) + Year + View, wired to every page below
# ===============================
top_l, top_r1, top_r2 = st.columns([5, 1.4, 1.4])
with top_r1:
    selected_year = st.selectbox("Period", years_sorted, index=len(years_sorted) - 1, label_visibility="collapsed")
with top_r2:
    view_mode = st.selectbox("View", ["Global View", "Country View"], label_visibility="collapsed")

st.session_state["view_mode"] = view_mode

# Country selector lives just under the top bar so "Country View" has a target
if view_mode == "Country View":
    selected_country = st.selectbox("Focus country", country_list, key="focus_country_select")
else:
    selected_country = st.session_state.get("focus_country_select", country_list[0])
st.session_state["focus_country"] = selected_country

page = {
    "🏠 Overview": "Global Overview",
    "📊 Economy": "Country Comparison",
    "💹 Markets": "Financial Stability",
    "🌍 Trade": "Trade Analytics",
    "🛡️ Risk Intelligence": "Global Risk Intelligence",
    "🤖 AI Predictions": "AI Prediction",
    "🧭 Country Analysis": "Country Clustering",
    "🗂️ Data Explorer": "Data Explorer",
}[nav_choice]

prev_year = previous_year(selected_year)

# ===============================
# PAGE 1 — GLOBAL OVERVIEW
# ===============================
if page == "Global Overview":
    hero("Global Economic Intelligence Platform", "AI-powered macroeconomic analytics")

    year_df = scoped_frame(selected_year)
    prev_df = scoped_frame(prev_year) if prev_year is not None else None
    scope_label = f"Global · {selected_year}" if view_mode == "Global View" else f"{selected_country} · {selected_year}"

    insight(
        f"<b>Executive summary:</b> Showing <b>{scope_label}</b>. "
        f"GDP, inflation, unemployment and growth below react instantly to the Period and View controls in the top bar."
    )

    def kpi_delta(col):
        if prev_df is None or prev_df.empty or year_df.empty:
            return None
        cur = year_df[col].mean()
        prv = prev_df[col].mean()
        if pd.isna(cur) or pd.isna(prv) or prv == 0:
            return None
        return (cur - prv) / abs(prv) * 100

    hist_scope = df if view_mode == "Global View" else df[df["Country"] == selected_country]
    hist_scope = hist_scope[hist_scope["Year"] <= selected_year]
    gdp_hist = hist_scope.groupby("Year", as_index=False)["GDP"].mean()
    inf_hist = hist_scope.groupby("Year", as_index=False)["Inflation"].mean()
    une_hist = hist_scope.groupby("Year", as_index=False)["Unemployment"].mean()
    grw_hist = hist_scope.groupby("Year", as_index=False)["GDP_Growth"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("🌐", "linear-gradient(135deg,#2563EB,#1D4ED8)", "Average GDP (Nominal)",
                  format_number(year_df["GDP"].mean(), prefix="$"), kpi_delta("GDP"),
                  f"vs {prev_year}" if prev_year else "No prior year", gdp_hist, "GDP", "rgb(59,130,246)")
    with c2:
        kpi_card("💲", "linear-gradient(135deg,#7C3AED,#5B21B6)", "Average Inflation (CPI)",
                  format_number(year_df["Inflation"].mean(), pct=True), kpi_delta("Inflation"),
                  f"vs {prev_year}" if prev_year else "No prior year", inf_hist, "Inflation", "rgb(129,140,248)")
    with c3:
        kpi_card("👥", "linear-gradient(135deg,#059669,#047857)", "Average Unemployment",
                  format_number(year_df["Unemployment"].mean(), pct=True), kpi_delta("Unemployment"),
                  f"vs {prev_year}" if prev_year else "No prior year", une_hist, "Unemployment", "rgb(34,197,94)")
    with c4:
        kpi_card("📈", "linear-gradient(135deg,#D97706,#B45309)", "Average GDP Growth",
                  format_number(year_df["GDP_Growth"].mean(), pct=True), kpi_delta("GDP_Growth"),
                  f"vs {prev_year}" if prev_year else "No prior year", grw_hist, "GDP_Growth", "rgb(245,158,11)")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(f"<div class='panel-title'>Global GDP Growth by Country</div><div class='panel-sub'>{selected_year}</div>", unsafe_allow_html=True)
        map_df = df[df["Year"] == selected_year]
        fig = px.choropleth(
            map_df, locations="Country", locationmode="country names", color="GDP_Growth",
            hover_name="Country", color_continuous_scale="Blues",
        )
        fig = style_fig(fig, height=430)
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True, coastlinecolor="#334155")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='panel-title'>Global GDP Growth Trend</div>", unsafe_allow_html=True)
        rng = st.radio("Range", ["1Y", "5Y", "10Y", "All"], index=1, horizontal=True, label_visibility="collapsed", key="gdp_trend_range")
        y0, y1 = range_years(rng, selected_year)
        trend_scope = df if view_mode == "Global View" else df[df["Country"] == selected_country]
        trend_df = trend_scope[(trend_scope["Year"] >= y0) & (trend_scope["Year"] <= y1)]
        trend_df = trend_df.groupby("Year", as_index=False)["GDP_Growth"].mean()
        fig = px.line(trend_df, x="Year", y="GDP_Growth", markers=True)
        fig = style_fig(fig, height=430)
        fig.update_traces(line_color="#3B82F6", fill="tozeroy", fillcolor="rgba(59,130,246,0.12)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left2, right2 = st.columns([1.05, 1])
    with left2:
        st.markdown(f"<div class='panel-title'>Regional Inflation Heatmap (CPI)</div><div class='panel-sub'>By region, up to {selected_year}</div>", unsafe_allow_html=True)
        heat_scope = df[df["Year"] <= selected_year]
        heat_scope = heat_scope[heat_scope["Year"] >= max(earliest_year, selected_year - 9)]
        pivot = heat_scope.pivot_table(index="Region", columns="Year", values="Inflation", aggfunc="mean")
        pivot = pivot.reindex(["North America", "South America", "Europe", "Asia", "Africa", "Oceania"]).dropna(how="all")
        fig = px.imshow(pivot, color_continuous_scale="RdBu_r", aspect="auto", labels=dict(color="Inflation (%)"))
        fig = style_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.markdown(f"<div class='panel-title'>Global Trade Analytics</div><div class='panel-sub'>{scope_label}</div>", unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.metric("Exports", format_number(year_df["Exports"].mean(), prefix="$"))
        with tc2:
            st.metric("Imports", format_number(year_df["Imports"].mean(), prefix="$"))
        with tc3:
            st.metric("Trade Balance", format_number(year_df["Trade_Balance"].mean(), prefix="$"))
        trade_scope = (df if view_mode == "Global View" else df[df["Country"] == selected_country])
        trade_scope = trade_scope[trade_scope["Year"] <= selected_year].groupby("Year", as_index=False)[["Exports", "Imports"]].mean()
        fig = px.line(trade_scope, x="Year", y=["Exports", "Imports"], markers=True)
        fig = style_fig(fig, height=290)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"<div class='panel-title'>AI Economic Prediction</div><div class='panel-sub'>{selected_year} snapshot</div>", unsafe_allow_html=True)
        growth_val = year_df["GDP_Growth"].mean()
        if pd.isna(growth_val):
            growth_val = 0.0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=growth_val,
            number={"suffix": "%", "font": {"size": 30, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [-5, 10], "tickcolor": "#64748B"},
                "bar": {"color": "#3B82F6"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [-5, 0], "color": "rgba(239,68,68,0.35)"},
                    {"range": [0, 4], "color": "rgba(245,158,11,0.30)"},
                    {"range": [4, 10], "color": "rgba(34,197,94,0.30)"},
                ],
            },
        ))
        fig = style_fig(fig, height=260)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Projected GDP growth · {scope_label}")

    with p2:
        st.markdown("<div class='panel-title'>Risk Intelligence</div>", unsafe_allow_html=True)
        counts = risk_level_counts()
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"<div class='risk-chip' style='background:rgba(5,150,105,0.12);border-color:rgba(5,150,105,0.35);'>"
                        f"<div class='rc-label'>Low Risk</div><div class='rc-count'>{counts['low']}</div></div>", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"<div class='risk-chip' style='background:rgba(217,119,6,0.12);border-color:rgba(217,119,6,0.35);'>"
                        f"<div class='rc-label'>Medium Risk</div><div class='rc-count'>{counts['medium']}</div></div>", unsafe_allow_html=True)
        rc3, rc4 = st.columns(2)
        with rc3:
            st.markdown(f"<div class='risk-chip' style='background:rgba(234,88,12,0.12);border-color:rgba(234,88,12,0.35);'>"
                        f"<div class='rc-label'>High Risk</div><div class='rc-count'>{counts['high']}</div></div>", unsafe_allow_html=True)
        with rc4:
            st.markdown(f"<div class='risk-chip' style='background:rgba(220,38,38,0.12);border-color:rgba(220,38,38,0.35);'>"
                        f"<div class='rc-label'>Crisis Risk</div><div class='rc-count'>{counts['crisis']}</div></div>", unsafe_allow_html=True)
        if sum(counts.values()) == 0:
            st.caption(f"⚠️ No Risk_Level values matched. Raw values in your data: {risk_level_raw_values()}")

    with p3:
        st.markdown(f"<div class='panel-title'>Top Countries by GDP</div><div class='panel-sub'>{selected_year}</div>", unsafe_allow_html=True)
        top10 = df[df["Year"] == selected_year][["Country", "GDP", "GDP_Growth"]].sort_values("GDP", ascending=False).head(10)
        top10["GDP"] = top10["GDP"].apply(lambda v: format_number(v, prefix="$"))
        top10["GDP_Growth"] = top10["GDP_Growth"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
        top10 = top10.rename(columns={"GDP": "GDP (Nominal)", "GDP_Growth": "GDP Growth"})
        st.dataframe(top10, use_container_width=True, hide_index=True)

# ===============================
# PAGE 2 — ECONOMY (country comparison)
# ===============================
elif page == "Country Comparison":
    hero("Economy", "Compare GDP, growth, inflation, unemployment and health score across countries")

    selected_countries = st.multiselect("Choose countries to compare", country_list, default=country_list[:3])
    compare_df = df[df["Country"].isin(selected_countries)].copy()
    compare_df = compare_df[compare_df["Year"] <= selected_year]

    metric = st.selectbox("Select metric", ["GDP", "GDP_Growth", "Inflation", "Unemployment", "Economic_Health_Score"])

    if metric == "Economic_Health_Score":
        score_plot = score_df.sort_values("Economic_Health_Score", ascending=False).head(20).sort_values("Economic_Health_Score")
        fig = px.bar(score_plot, y="Country", x="Economic_Health_Score", orientation="h", title="Top Countries by Economic Health Score")
    else:
        fig = px.line(compare_df, x="Year", y=metric, color="Country", markers=True, title=f"{metric.replace('_', ' ')} Comparison (up to {selected_year})")

    fig = style_fig(fig, height=460)
    st.plotly_chart(fig, use_container_width=True)

    ranking_df = df[df["Year"] == selected_year][["Country", "GDP"]].sort_values("GDP", ascending=False).head(20)
    styled_dataframe(ranking_df, {"GDP": "{:,.0f}"})

# ===============================
# PAGE 3 — MARKETS (financial stability)
# ===============================
elif page == "Financial Stability":
    hero("Markets", "Reserves, stock market indicators, and exchange rate conditions")

    country_fin = df[(df["Country"] == selected_country) & (df["Year"] <= selected_year)].copy()
    latest_country = country_fin[country_fin["Year"] == country_fin["Year"].max()]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("💰", "linear-gradient(135deg,#2563EB,#1D4ED8)", "Total Reserves",
                  format_number(latest_country["Total_Reserves"].mean(), prefix="$"), None, selected_country)
    with c2:
        kpi_card("📊", "linear-gradient(135deg,#7C3AED,#5B21B6)", "Stock Market",
                  format_number(latest_country["Stock_Market"].mean()), None, selected_country)
    with c3:
        kpi_card("💱", "linear-gradient(135deg,#059669,#047857)", "Exchange Rate",
                  format_number(latest_country["Exchange_Rate"].mean()), None, selected_country)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(country_fin, x="Year", y="Total_Reserves", markers=True, title=f"Total Reserves — {selected_country}")
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(country_fin, x="Year", y="Stock_Market", markers=True, title=f"Stock Market — {selected_country}")
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(country_fin, x="Year", y="Exchange_Rate", markers=True, title=f"Exchange Rate — {selected_country}")
    fig = style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 4 — TRADE
# ===============================
elif page == "Trade Analytics":
    hero("Trade", "Exports, imports, and trade balance to gauge external sector strength")

    country_trade = df[(df["Country"] == selected_country) & (df["Year"] <= selected_year)].copy()
    latest_country = country_trade[country_trade["Year"] == country_trade["Year"].max()]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("📤", "linear-gradient(135deg,#2563EB,#1D4ED8)", "Latest Exports",
                  format_number(latest_country["Exports"].mean(), prefix="$"), None, selected_country)
    with c2:
        kpi_card("📥", "linear-gradient(135deg,#7C3AED,#5B21B6)", "Latest Imports",
                  format_number(latest_country["Imports"].mean(), prefix="$"), None, selected_country)
    with c3:
        kpi_card("⚖️", "linear-gradient(135deg,#059669,#047857)", "Trade Balance",
                  format_number(latest_country["Trade_Balance"].mean(), prefix="$"), None, selected_country)

    tab1, tab2 = st.tabs(["Country trend", "Global ranking"])
    with tab1:
        fig = px.line(country_trade, x="Year", y=["Exports", "Imports"], markers=True, title=f"Exports vs Imports — {selected_country}")
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(country_trade, x="Year", y="Trade_Balance", title=f"Trade Balance — {selected_country}")
        fig = style_fig(fig)
        fig.update_traces(marker_color="#3B82F6")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        year_slice = df[df["Year"] == selected_year]
        with col1:
            st.markdown(f"### Top Trade Surplus — {selected_year}")
            surplus = year_slice[["Country", "Trade_Balance"]].sort_values("Trade_Balance", ascending=False).head(10)
            styled_dataframe(surplus, {"Trade_Balance": "{:,.0f}"})
        with col2:
            st.markdown(f"### Top Trade Deficit — {selected_year}")
            deficit = year_slice[["Country", "Trade_Balance"]].sort_values("Trade_Balance", ascending=True).head(10)
            styled_dataframe(deficit, {"Trade_Balance": "{:,.0f}"})

# ===============================
# PAGE 5 — RISK INTELLIGENCE
# ===============================
elif page == "Global Risk Intelligence":
    hero("Risk Intelligence", "High-risk countries, anomaly patterns, and crisis warning signals")

    risk_year_df, _, anomaly_year_df, _ = get_year_filtered_data(selected_year)

    counts = {
    "low": (risk_year_df["Risk_Level"] == "Low Risk").sum(),
    "medium": (risk_year_df["Risk_Level"] == "Medium Risk").sum(),
    "high": (risk_year_df["Risk_Level"] == "High Risk").sum(),
    "crisis": (risk_year_df["Risk_Level"] == "Crisis Risk").sum(),
    }
    rc1, rc2, rc3, rc4 = st.columns(4)
    for col, label, bucket, bg, border in [
        (rc1, "Low Risk", "low", "rgba(5,150,105,0.12)", "rgba(5,150,105,0.35)"),
        (rc2, "Medium Risk", "medium", "rgba(217,119,6,0.12)", "rgba(217,119,6,0.35)"),
        (rc3, "High Risk", "high", "rgba(234,88,12,0.12)", "rgba(234,88,12,0.35)"),
        (rc4, "Crisis Risk", "crisis", "rgba(220,38,38,0.12)", "rgba(220,38,38,0.35)"),
    ]:
        with col:
            st.markdown(f"<div class='risk-chip' style='background:{bg};border-color:{border};'>"
                        f"<div class='rc-label'>{label}</div><div class='rc-count'>{counts[bucket]}</div>"
                        f"<div class='rc-label'>Countries</div></div>", unsafe_allow_html=True)
    if sum(counts.values()) == 0:
        st.caption(f"⚠️ No Risk_Level values matched. Raw values in your data: {risk_level_raw_values()}")

    top_risk = risk_year_df.sort_values(
    "Global_Risk_Score",
    ascending=False
    ).head(15)
    fig = px.bar(top_risk, y="Country", x="Global_Risk_Score", color="Risk_Level", orientation="h", title="Top Risky Countries")
    fig = style_fig(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

    risk_cols = ["Country", "Global_Risk_Score", "Risk_Level"]
    if "Crisis_Alert" in risk_df.columns:
        risk_cols.append("Crisis_Alert")
    styled_dataframe(
    risk_year_df.sort_values(
        "Global_Risk_Score",
        ascending=False
    ).head(15)[risk_cols],
    {"Global_Risk_Score": "{:.2f}"}
    )

    st.markdown("### Anomaly Detection")
    anomaly_year = anomaly_df["Year"].max() if selected_year not in anomaly_df["Year"].unique() else selected_year
    latest_anomaly = anomaly_year_df
    fig = px.scatter(latest_anomaly, x="GDP_Growth", y="Inflation", color="Anomaly_Label", hover_name="Country",
                      title=f"Economic Anomaly Detection — {anomaly_year}")
    fig = style_fig(fig, height=480)
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 6 — AI PREDICTIONS
# ===============================
elif page == "AI Prediction":
    hero("AI Predictions", "Inflation prediction, recession probability, and AI-based risk signals")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("<div class='panel-title'>Projected GDP Growth</div>", unsafe_allow_html=True)
        yr_df = scoped_frame(selected_year)
        growth_val = yr_df["GDP_Growth"].mean() if not yr_df.empty else 0.0
        if pd.isna(growth_val):
            growth_val = 0.0
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=growth_val, number={"suffix": "%", "font": {"size": 34, "color": "#F8FAFC"}},
            gauge={"axis": {"range": [-5, 10], "tickcolor": "#64748B"}, "bar": {"color": "#3B82F6"},
                   "bgcolor": "rgba(0,0,0,0)",
                   "steps": [{"range": [-5, 0], "color": "rgba(239,68,68,0.35)"},
                             {"range": [0, 4], "color": "rgba(245,158,11,0.30)"},
                             {"range": [4, 10], "color": "rgba(34,197,94,0.30)"}]},
        ))
        fig = style_fig(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{'Global' if view_mode == 'Global View' else selected_country} · {selected_year}")

    with right:
        st.markdown("### Inflation Prediction Input")
        col1, col2, col3 = st.columns(3)
        with col1:
            gdp = st.number_input("GDP", value=float(latest_df["GDP"].mean()))
            gdp_growth = st.number_input("GDP Growth", value=float(latest_df["GDP_Growth"].mean()))
            unemployment = st.number_input("Unemployment", value=float(latest_df["Unemployment"].mean()))
        with col2:
            trade_balance = st.number_input("Trade Balance", value=float(latest_df["Trade_Balance"].mean()))
            exchange_rate = st.number_input("Exchange Rate", value=float(latest_df["Exchange_Rate"].mean()))
            stock_growth = st.number_input("Stock Growth", value=float(latest_df["Stock_Growth"].mean()))
        with col3:
            inflation_lag = st.number_input("Inflation Lag", value=float(df["Inflation_Lag"].mean()))
            gdp_growth_lag = st.number_input("GDP Growth Lag", value=float(df["GDP_Growth_Lag"].mean()))
            unemployment_lag = st.number_input("Unemployment Lag", value=float(df["Unemployment_Lag"].mean()))

        try:
            inflation_model = joblib.load("models/inflation_model.pkl")
        except Exception:
            inflation_model = joblib.load("inflation_model.pkl")

        input_data = pd.DataFrame({
            "GDP": [gdp], "GDP_Growth": [gdp_growth], "Unemployment": [unemployment],
            "Trade_Balance": [trade_balance], "Exchange_Rate": [exchange_rate], "Stock_Growth": [stock_growth],
            "Inflation_Lag": [inflation_lag], "GDP_Growth_Lag": [gdp_growth_lag], "Unemployment_Lag": [unemployment_lag],
        })

        if st.button("Predict Inflation"):
            prediction = inflation_model.predict(input_data)[0]
            st.success(f"Predicted Inflation: {prediction:.2f}%")

    st.markdown("### Recession Probability")
    selected_rec = recession_df[recession_df["Country"] == selected_country].sort_values("Year")
    selected_rec = selected_rec[selected_rec["Year"] <= selected_year]
    if "Recession_Probability" in selected_rec.columns:
        fig = px.line(selected_rec, x="Year", y="Recession_Probability", markers=True, title=f"Recession Probability — {selected_country}")
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 7 — COUNTRY ANALYSIS (clustering)
# ===============================
elif page == "Country Clustering":
    hero("Country Analysis", "Segment countries using KMeans clustering and PCA-based economic positioning")
    _, cluster_year_df, _, _ = get_year_filtered_data(selected_year)

    fig = px.scatter(cluster_year_df, x="PCA1", y="PCA2", color="Cluster", hover_name="Country", title="Country Economic Segmentation")
    fig = style_fig(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster Profile")
    cluster_features = ["GDP", "GDP_Growth", "Inflation", "Unemployment", "Trade_Balance", "Total_Reserves", "Stock_Growth"]
    cluster_profile = cluster_year_df.groupby("Cluster")[cluster_features].mean()
    styled_dataframe(cluster_profile)

    st.markdown("### Countries by Cluster")
    selected_cluster = st.selectbox(
    "Select cluster",
    sorted(cluster_year_df["Cluster"].unique())
    )
    st.dataframe(cluster_year_df[
    cluster_year_df["Cluster"] == selected_cluster][["Country", "Cluster"]], use_container_width=True)

# ===============================
# PAGE 8 — DATA EXPLORER
# ===============================
elif page == "Data Explorer":
    hero("Data Explorer", f"Raw records for {'all countries' if view_mode == 'Global View' else selected_country} up to {selected_year}")

    explorer_df = df[df["Year"] <= selected_year]
    if view_mode == "Country View":
        explorer_df = explorer_df[explorer_df["Country"] == selected_country]

    search = st.text_input("Filter by country name (optional)")
    if search:
        explorer_df = explorer_df[explorer_df["Country"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(explorer_df):,} rows")
    st.dataframe(explorer_df, use_container_width=True)
    st.download_button("Download CSV", explorer_df.to_csv(index=False).encode("utf-8"), file_name="geip_export.csv")
