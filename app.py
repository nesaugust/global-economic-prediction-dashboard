import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import joblib

st.set_page_config(
    page_title="Global Economic Intelligence Platform",
    page_icon="🌍",
    layout="wide",
)

# ===============================
# SAFE PLOTLY TEMPLATE
# ===============================
pio.templates.default = "plotly_dark"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 25% 10%, rgba(37,99,235,0.18), transparent 28%),
        radial-gradient(circle at 85% 5%, rgba(14,165,233,0.10), transparent 26%),
        linear-gradient(135deg, #030712 0%, #07111f 42%, #0b1220 100%);
    color: #F8FAFC;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 4rem;
    max-width: 1320px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #030712 0%, #07111f 55%, #020617 100%);
    border-right: 1px solid rgba(148,163,184,0.14);
}

section[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

section[data-testid="stSidebar"] h1 {
    font-size: 25px !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 14px;
    padding: 0.45rem 0.55rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(37,99,235,0.18);
}

/* Text */
h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
    letter-spacing: -1.2px;
    color: #F8FAFC !important;
}

h2, h3 {
    color: #F8FAFC !important;
    font-weight: 750 !important;
}

p, span, label {
    color: #CBD5E1 !important;
}

/* Hero */
.hero-card {
    background:
        radial-gradient(circle at 85% 20%, rgba(59,130,246,0.26), transparent 34%),
        linear-gradient(135deg, #172554 0%, #10203d 45%, #111827 100%);
    border: 1px solid rgba(96,165,250,0.28);
    border-radius: 28px;
    padding: 34px 38px;
    margin-bottom: 22px;
    box-shadow: 0 24px 65px rgba(2,6,23,0.34);
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1.1px;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 15px;
    color: #BFDBFE;
    line-height: 1.7;
    max-width: 920px;
}

/* Executive summary */
.insight-box {
    background:
        linear-gradient(135deg, rgba(15,23,42,0.92), rgba(15,23,42,0.72));
    border: 1px solid rgba(96,165,250,0.24);
    border-left: 4px solid #3B82F6;
    padding: 18px 22px;
    border-radius: 18px;
    margin: 14px 0 22px 0;
    color: #DBEAFE;
    line-height: 1.6;
    box-shadow: 0 14px 32px rgba(0,0,0,0.20);
}

/* KPI cards */
.metric-card {
    background:
        linear-gradient(145deg, rgba(30,41,59,0.92), rgba(15,23,42,0.92));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 24px 24px;
    box-shadow: 0 18px 42px rgba(0,0,0,0.26);
    min-height: 125px;
}

.metric-card:hover {
    border-color: rgba(96,165,250,0.38);
    transform: translateY(-2px);
    transition: all 0.18s ease;
}

.metric-title {
    font-size: 12px;
    color: #93C5FD;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    margin-top: 12px;
    color: #FFFFFF;
}

.metric-note {
    font-size: 12px;
    color: #94A3B8;
    margin-top: 8px;
}

/* Tabs container effect */
div[data-testid="stTabs"] {
    background: rgba(15,23,42,0.42);
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 22px;
    padding: 10px 16px 16px 16px;
    margin-top: 10px;
}

button[data-baseweb="tab"] {
    font-weight: 800;
    color: #CBD5E1 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #60A5FA !important;
}

/* Inputs */
div[data-baseweb="select"] > div,
input {
    background-color: #0F172A !important;
    border: 1px solid rgba(148,163,184,0.26) !important;
    border-radius: 14px !important;
    color: #F8FAFC !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white !important;
    border: none;
    border-radius: 14px;
    padding: 0.72rem 1.2rem;
    font-weight: 800;
    box-shadow: 0 12px 28px rgba(37,99,235,0.24);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF);
}

/* Plotly chart card */
.js-plotly-plot {
    background:
        linear-gradient(145deg, rgba(15,23,42,0.94), rgba(17,24,39,0.90)) !important;
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.14);
    box-shadow: 0 18px 45px rgba(0,0,0,0.24);
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    background: #0F172A !important;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.16);
}

div[data-testid="stDataFrame"] thead tr th {
    background: #1E293B !important;
    color: #F8FAFC !important;
    font-weight: 800 !important;
}

div[data-testid="stDataFrame"] tbody tr {
    background: #0F172A !important;
}

div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: #111827 !important;
}

div[data-testid="stDataFrame"] td {
    color: #CBD5E1 !important;
}

div[data-testid="stDataFrame"] tbody tr:hover {
    background: #1F2937 !important;
}

/* Reduce default white top gap */
header[data-testid="stHeader"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
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

if "Crisis_Alert" not in risk_df.columns:
    risk_df["Crisis_Alert"] = np.where(
        risk_df["Risk_Level"].astype(str).str.lower().eq("high"),
        "Crisis Warning",
        "Normal",
    )

latest_year = int(df["Year"].max())
latest_df = df[df["Year"] == latest_year].copy()
country_list = sorted(df["Country"].dropna().unique())

# ===============================
# HELPERS
# ===============================
def format_number(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def kpi_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(text):
    st.markdown(f"<div class='insight-box'>{text}</div>", unsafe_allow_html=True)


def style_fig(fig, height=430):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.18)",
        font=dict(
            family="Inter, Arial",
            size=13,
            color="#CBD5E1",
        ),
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=18, color="#F8FAFC"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            font=dict(color="#CBD5E1"),
        ),
        margin=dict(l=42, r=28, t=72, b=46),
        hovermode="x unified",
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
        
    fig.update_layout(
        colorway=["#3B82F6", "#38BDF8", "#818CF8", "#22C55E", "#F59E0B", "#EF4444"]
    )

    return fig


def styled_dataframe(dataframe, formats=None):
    if formats:
        st.dataframe(dataframe.style.format(formats), use_container_width=True)
    else:
        st.dataframe(dataframe, use_container_width=True)


# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Global Economy AI")
st.sidebar.caption("Executive economic intelligence dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Global Overview",
        "Country Comparison",
        "Trade Analytics",
        "Financial Stability",
        "AI Prediction",
        "Global Risk Intelligence",
        "Country Clustering",
    ],
)

st.sidebar.markdown("---")

selected_year = st.sidebar.selectbox(
    "Select year",
    sorted(df["Year"].dropna().unique(), reverse=True),
)

selected_country = st.sidebar.selectbox(
    "Select country",
    country_list,
)

# ===============================
# PAGE 1 — GLOBAL OVERVIEW
# ===============================
if page == "Global Overview":
    hero(
        "Global Economic Intelligence Platform",
        "Monitor macroeconomic performance, inflation dynamics, unemployment pressure, trade conditions, and AI-powered risk indicators across countries.",
    )

    year_df = df[df["Year"] == selected_year].copy()

    insight(
        f"<b>Executive summary:</b> In {selected_year}, this dashboard tracks GDP, inflation, unemployment, and GDP growth across available countries."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Average GDP", format_number(year_df["GDP"].mean()), f"Selected year: {selected_year}")
    with c2:
        kpi_card("Average Inflation", f"{year_df['Inflation'].mean():.2f}%", "Average across countries")
    with c3:
        kpi_card("Average Unemployment", f"{year_df['Unemployment'].mean():.2f}%", "Labour market pressure")
    with c4:
        kpi_card("Average GDP Growth", f"{year_df['GDP_Growth'].mean():.2f}%", "Growth momentum")

    tab1, tab2, tab3 = st.tabs(["Global map", "Trends", "Rankings"])

    with tab1:
        fig = px.choropleth(
            year_df,
            locations="Country",
            locationmode="country names",
            color="GDP_Growth",
            hover_name="Country",
            title=f"Global GDP Growth Map — {selected_year}",
            color_continuous_scale="Blues",
        )
        fig = style_fig(fig, height=520)
        fig.update_geos(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#334155",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            gdp_trend = df.groupby("Year", as_index=False)["GDP"].mean()
            fig = px.line(gdp_trend, x="Year", y="GDP", markers=True, title="Average Global GDP Trend")
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            inflation_trend = df.groupby("Year", as_index=False)["Inflation"].mean()
            fig = px.line(inflation_trend, x="Year", y="Inflation", markers=True, title="Average Global Inflation Trend")
            fig = style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        unemployment_trend = df.groupby("Year", as_index=False)["Unemployment"].mean()
        fig = px.line(unemployment_trend, x="Year", y="Unemployment", markers=True, title="Average Global Unemployment Trend")
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        ranking_df = latest_df[["Country", "GDP"]].sort_values("GDP", ascending=False).head(15).sort_values("GDP")
        fig = px.bar(
            ranking_df,
            y="Country",
            x="GDP",
            orientation="h",
            title=f"Top 15 Countries by GDP — {latest_year}",
        )
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 2 — COUNTRY COMPARISON
# ===============================
elif page == "Country Comparison":
    hero(
        "Country Comparison",
        "Compare countries across GDP, growth, inflation, unemployment, and economic health score.",
    )

    selected_countries = st.multiselect(
        "Choose countries to compare",
        country_list,
        default=country_list[:3],
    )

    compare_df = df[df["Country"].isin(selected_countries)].copy()

    metric = st.selectbox(
        "Select metric",
        ["GDP", "GDP_Growth", "Inflation", "Unemployment", "Economic_Health_Score"],
    )

    if metric == "Economic_Health_Score":
        score_plot = score_df.sort_values("Economic_Health_Score", ascending=False).head(20).sort_values("Economic_Health_Score")
        fig = px.bar(
            score_plot,
            y="Country",
            x="Economic_Health_Score",
            orientation="h",
            title="Top Countries by Economic Health Score",
        )
    else:
        fig = px.line(
            compare_df,
            x="Year",
            y=metric,
            color="Country",
            markers=True,
            title=f"{metric.replace('_', ' ')} Comparison",
        )

    fig = style_fig(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    ranking_df = latest_df[["Country", "GDP"]].sort_values("GDP", ascending=False).head(20)
    styled_dataframe(ranking_df, {"GDP": "{:,.0f}"})

# ===============================
# PAGE 3 — TRADE ANALYTICS
# ===============================
elif page == "Trade Analytics":
    hero(
        "Trade Analytics",
        "Analyze exports, imports, and trade balance to understand external sector strength.",
    )

    country_trade = df[df["Country"] == selected_country].copy()
    latest_country = country_trade[country_trade["Year"] == country_trade["Year"].max()]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Latest Exports", format_number(latest_country["Exports"].mean()), selected_country)
    with c2:
        kpi_card("Latest Imports", format_number(latest_country["Imports"].mean()), selected_country)
    with c3:
        kpi_card("Trade Balance", format_number(latest_country["Trade_Balance"].mean()), selected_country)

    tab1, tab2 = st.tabs(["Country trend", "Global ranking"])

    with tab1:
        fig = px.line(
            country_trade,
            x="Year",
            y=["Exports", "Imports"],
            markers=True,
            title=f"Exports vs Imports — {selected_country}",
        )
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            country_trade,
            x="Year",
            y="Trade_Balance",
            title=f"Trade Balance — {selected_country}",
        )
        fig = style_fig(fig)
        fig.update_traces(marker_color="#3B82F6")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top Trade Surplus")
            surplus = latest_df[["Country", "Trade_Balance"]].sort_values("Trade_Balance", ascending=False).head(10)
            styled_dataframe(surplus, {"Trade_Balance": "{:,.0f}"})

        with col2:
            st.markdown("### Top Trade Deficit")
            deficit = latest_df[["Country", "Trade_Balance"]].sort_values("Trade_Balance", ascending=True).head(10)
            styled_dataframe(deficit, {"Trade_Balance": "{:,.0f}"})

# ===============================
# PAGE 4 — FINANCIAL STABILITY
# ===============================
elif page == "Financial Stability":
    hero(
        "Financial Stability",
        "Monitor reserves, stock market indicators, and exchange rate conditions.",
    )

    country_fin = df[df["Country"] == selected_country].copy()
    latest_country = country_fin[country_fin["Year"] == country_fin["Year"].max()]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Total Reserves", format_number(latest_country["Total_Reserves"].mean()), selected_country)
    with c2:
        kpi_card("Stock Market", format_number(latest_country["Stock_Market"].mean()), selected_country)
    with c3:
        kpi_card("Exchange Rate", format_number(latest_country["Exchange_Rate"].mean()), selected_country)

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
# PAGE 5 — AI PREDICTION
# ===============================
elif page == "AI Prediction":
    hero(
        "AI Prediction",
        "Run inflation prediction, monitor recession probability, and explore AI-based economic risk signals.",
    )

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
        "GDP": [gdp],
        "GDP_Growth": [gdp_growth],
        "Unemployment": [unemployment],
        "Trade_Balance": [trade_balance],
        "Exchange_Rate": [exchange_rate],
        "Stock_Growth": [stock_growth],
        "Inflation_Lag": [inflation_lag],
        "GDP_Growth_Lag": [gdp_growth_lag],
        "Unemployment_Lag": [unemployment_lag],
    })

    if st.button("Predict Inflation"):
        prediction = inflation_model.predict(input_data)[0]
        st.success(f"Predicted Inflation: {prediction:.2f}%")

    st.markdown("### Recession Probability")

    selected_rec = recession_df[recession_df["Country"] == selected_country].sort_values("Year")

    if "Recession_Probability" in selected_rec.columns:
        fig = px.line(
            selected_rec,
            x="Year",
            y="Recession_Probability",
            markers=True,
            title=f"Recession Probability — {selected_country}",
        )
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 6 — GLOBAL RISK INTELLIGENCE
# ===============================
elif page == "Global Risk Intelligence":
    hero(
        "Global Risk Intelligence",
        "Identify high-risk countries, anomaly patterns, and crisis warning signals.",
    )

    top_risk = risk_df.sort_values("Global_Risk_Score", ascending=False).head(15).sort_values("Global_Risk_Score")

    fig = px.bar(
        top_risk,
        y="Country",
        x="Global_Risk_Score",
        color="Risk_Level",
        orientation="h",
        title="Top Risky Countries",
    )
    fig = style_fig(fig, height=540)
    st.plotly_chart(fig, use_container_width=True)

    risk_cols = ["Country", "Global_Risk_Score", "Risk_Level"]
    if "Crisis_Alert" in risk_df.columns:
        risk_cols.append("Crisis_Alert")

    styled_dataframe(
        risk_df.sort_values("Global_Risk_Score", ascending=False).head(15)[risk_cols],
        {"Global_Risk_Score": "{:.2f}"},
    )

    st.markdown("### Anomaly Detection")

    latest_anomaly = anomaly_df[anomaly_df["Year"] == anomaly_df["Year"].max()]

    fig = px.scatter(
        latest_anomaly,
        x="GDP_Growth",
        y="Inflation",
        color="Anomaly_Label",
        hover_name="Country",
        title="Economic Anomaly Detection",
    )
    fig = style_fig(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 7 — COUNTRY CLUSTERING
# ===============================
elif page == "Country Clustering":
    hero(
        "Country Clustering",
        "Segment countries using KMeans clustering and PCA-based economic positioning.",
    )

    fig = px.scatter(
        cluster_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_name="Country",
        title="Country Economic Segmentation",
    )
    fig = style_fig(fig, height=540)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster Profile")

    cluster_features = [
        "GDP",
        "GDP_Growth",
        "Inflation",
        "Unemployment",
        "Trade_Balance",
        "Total_Reserves",
        "Stock_Growth",
    ]

    cluster_profile = cluster_df.groupby("Cluster")[cluster_features].mean()
    styled_dataframe(cluster_profile)

    st.markdown("### Countries by Cluster")

    selected_cluster = st.selectbox(
        "Select cluster",
        sorted(cluster_df["Cluster"].unique()),
    )

    st.dataframe(
        cluster_df[cluster_df["Cluster"] == selected_cluster][["Country", "Cluster"]],
        use_container_width=True,
    )
