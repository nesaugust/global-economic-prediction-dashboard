import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import plotly.io as pio

pio.templates["professional_dark"] = pio.templates["plotly_dark"]
pio.templates.default = "professional_dark"

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Global Economic Intelligence Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #111827 45%, #1E293B 100%);
    color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1E293B 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
}

h2, h3 {
    color: #E5E7EB !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 800;
}

.metric-card {
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    padding: 24px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 18px 40px rgba(0,0,0,0.28);
    border: 1px solid rgba(255,255,255,0.15);
}

.metric-title {
    font-size: 14px;
    color: #DBEAFE;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.section-card {
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.22);
    padding: 24px;
    border-radius: 24px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.25);
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

.stButton > button {
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.3rem;
    font-weight: 700;
}

.stSelectbox, .stNumberInput, .stMultiSelect {
    border-radius: 14px;
}

hr {
    border-color: rgba(255,255,255,0.12);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("processed_global_economy_data.csv")
    risk_df = pd.read_csv("risk_score_data.csv")
    score_df = pd.read_csv("economic_score_data.csv")
    cluster_df = pd.read_csv("country_cluster_data.csv")
    anomaly_df = pd.read_csv("anomaly_detection_data.csv")
    recession_df = pd.read_csv("recession_probability_data.csv")
    return df, risk_df, score_df, cluster_df, anomaly_df, recession_df

df, risk_df, score_df, cluster_df, anomaly_df, recession_df = load_data()

# Fix missing Crisis_Alert column
if "Crisis_Alert" not in risk_df.columns:
    risk_df["Crisis_Alert"] = np.where(
        risk_df["Risk_Level"].astype(str).str.lower().eq("high"),
        "Crisis Warning",
        "Normal"
    )

latest_year = df["Year"].max()
latest_df = df[df["Year"] == latest_year]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Global Economy AI")
st.sidebar.caption("Economic Intelligence Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Global Overview",
        "Country Comparison",
        "Trade Analytics",
        "Financial Stability",
        "AI Prediction",
        "Global Risk Intelligence",
        "Country Clustering"
    ]
)

st.sidebar.markdown("---")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].dropna().unique(), reverse=True)
)

country_list = sorted(df["Country"].dropna().unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    country_list
)

# ===============================
# HELPER FUNCTIONS
# ===============================
def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def format_number(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return f"{value:.2f}"

# ===============================
# PAGE 1 — GLOBAL OVERVIEW
# ===============================
if page == "Global Overview":
    st.title("Global Economic Overview")
    st.caption("Track global GDP, inflation, unemployment, and macroeconomic trends.")

    year_df = df[df["Year"] == selected_year]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Average GDP", format_number(year_df["GDP"].mean()))
    with c2:
        kpi_card("Average Inflation", f"{year_df['Inflation'].mean():.2f}%")
    with c3:
        kpi_card("Average Unemployment", f"{year_df['Unemployment'].mean():.2f}%")
    with c4:
        kpi_card("Average GDP Growth", f"{year_df['GDP_Growth'].mean():.2f}%")

    st.markdown("### Global Trends")

    col1, col2 = st.columns(2)

    with col1:
        gdp_trend = df.groupby("Year")["GDP"].mean().reset_index()
        fig = px.line(
            gdp_trend,
            x="Year",
            y="GDP",
            title="Average Global GDP Trend",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        inflation_trend = df.groupby("Year")["Inflation"].mean().reset_index()
        fig = px.line(
            inflation_trend,
            x="Year",
            y="Inflation",
            title="Average Global Inflation Trend",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    unemployment_trend = df.groupby("Year")["Unemployment"].mean().reset_index()
    fig = px.line(
        unemployment_trend,
        x="Year",
        y="Unemployment",
        title="Average Global Unemployment Trend",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 2 — COUNTRY COMPARISON
# ===============================
elif page == "Country Comparison":
    st.title("Country Comparison")
    st.caption("Compare countries based on GDP, inflation, unemployment, and economic score.")

    selected_countries = st.multiselect(
        "Choose countries to compare",
        country_list,
        default=country_list[:3]
    )

    compare_df = df[df["Country"].isin(selected_countries)]

    metric = st.selectbox(
        "Select metric",
        ["GDP", "GDP_Growth", "Inflation", "Unemployment", "Economic_Health_Score"]
    )

    if metric == "Economic_Health_Score":
        fig = px.bar(
            score_df.sort_values("Economic_Health_Score", ascending=False).head(20),
            x="Country",
            y="Economic_Health_Score",
            title="Top Countries by Economic Health Score"
        )
    else:
        fig = px.line(
            compare_df,
            x="Year",
            y=metric,
            color="Country",
            markers=True,
            title=f"{metric} Comparison"
        )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### GDP Ranking")

    ranking_df = latest_df[["Country", "GDP"]].sort_values("GDP", ascending=False).head(20)

    fig = px.bar(
        ranking_df,
        x="Country",
        y="GDP",
        title=f"Top 20 Countries by GDP in {latest_year}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(ranking_df, use_container_width=True)

# ===============================
# PAGE 3 — TRADE ANALYTICS
# ===============================
elif page == "Trade Analytics":
    st.title("Trade Analytics")
    st.caption("Analyze exports, imports, and trade balance by country.")

    country_trade = df[df["Country"] == selected_country]

    c1, c2, c3 = st.columns(3)

    latest_country = country_trade[country_trade["Year"] == country_trade["Year"].max()]

    with c1:
        kpi_card("Latest Exports", format_number(latest_country["Exports"].mean()))
    with c2:
        kpi_card("Latest Imports", format_number(latest_country["Imports"].mean()))
    with c3:
        kpi_card("Trade Balance", format_number(latest_country["Trade_Balance"].mean()))

    fig = px.line(
        country_trade,
        x="Year",
        y=["Exports", "Imports"],
        title=f"Exports vs Imports — {selected_country}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        country_trade,
        x="Year",
        y="Trade_Balance",
        title=f"Trade Balance — {selected_country}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Trade Surplus & Deficit")

    col1, col2 = st.columns(2)

    with col1:
        surplus = latest_df[["Country", "Trade_Balance"]].sort_values("Trade_Balance", ascending=False).head(10)
        st.dataframe(surplus, use_container_width=True)

    with col2:
        deficit = latest_df[["Country", "Trade_Balance"]].sort_values("Trade_Balance", ascending=True).head(10)
        st.dataframe(deficit, use_container_width=True)

# ===============================
# PAGE 4 — FINANCIAL STABILITY
# ===============================
elif page == "Financial Stability":
    st.title("Financial Stability")
    st.caption("Monitor reserves, stock market, and exchange rate conditions.")

    country_fin = df[df["Country"] == selected_country]

    c1, c2, c3 = st.columns(3)

    latest_country = country_fin[country_fin["Year"] == country_fin["Year"].max()]

    with c1:
        kpi_card("Total Reserves", format_number(latest_country["Total_Reserves"].mean()))
    with c2:
        kpi_card("Stock Market", format_number(latest_country["Stock_Market"].mean()))
    with c3:
        kpi_card("Exchange Rate", format_number(latest_country["Exchange_Rate"].mean()))

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            country_fin,
            x="Year",
            y="Total_Reserves",
            title=f"Total Reserves — {selected_country}",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            country_fin,
            x="Year",
            y="Stock_Market",
            title=f"Stock Market — {selected_country}",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        country_fin,
        x="Year",
        y="Exchange_Rate",
        title=f"Exchange Rate — {selected_country}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 5 — AI PREDICTION
# ===============================
elif page == "AI Prediction":
    st.title("AI Prediction")
    st.caption("Inflation prediction, recession probability, and 3-year forecast.")

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
    except:
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
        "Unemployment_Lag": [unemployment_lag]
    })

    if st.button("Predict Inflation"):
        prediction = inflation_model.predict(input_data)[0]
        st.success(f"Predicted Inflation: {prediction:.2f}%")

    st.markdown("### Recession Probability")

    selected_rec = recession_df[
        recession_df["Country"] == selected_country
    ].sort_values("Year")

    if "Recession_Probability" in selected_rec.columns:
        fig = px.line(
            selected_rec,
            x="Year",
            y="Recession_Probability",
            title=f"Recession Probability — {selected_country}",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 6 — GLOBAL RISK INTELLIGENCE
# ===============================
elif page == "Global Risk Intelligence":
    st.title("Global Risk Intelligence")
    st.caption("Identify high-risk countries, anomalies, and crisis alerts.")

    st.markdown("### Top Risky Countries")

    top_risk = risk_df.sort_values("Global_Risk_Score", ascending=False).head(15)

    fig = px.bar(
        top_risk,
        x="Country",
        y="Global_Risk_Score",
        color="Risk_Level",
        title="Top Risky Countries"
    )
    st.plotly_chart(fig, use_container_width=True)

    risk_cols = ["Country", "Global_Risk_Score", "Risk_Level"]

    if "Crisis_Alert" in top_risk.columns:
        risk_cols.append("Crisis_Alert")

    st.dataframe(
        top_risk[risk_cols],
        use_container_width=True
    )

    st.markdown("### Anomaly Detection")

    latest_anomaly = anomaly_df[anomaly_df["Year"] == anomaly_df["Year"].max()]

    fig = px.scatter(
        latest_anomaly,
        x="GDP_Growth",
        y="Inflation",
        color="Anomaly_Label",
        hover_name="Country",
        title="Economic Anomaly Detection"
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE 7 — COUNTRY CLUSTERING
# ===============================
elif page == "Country Clustering":
    st.title("Country Clustering")
    st.caption("Economic segmentation using KMeans and PCA visualization.")

    fig = px.scatter(
        cluster_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_name="Country",
        title="Country Economic Segmentation"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster Profile")

    cluster_features = [
        "GDP",
        "GDP_Growth",
        "Inflation",
        "Unemployment",
        "Trade_Balance",
        "Total_Reserves",
        "Stock_Growth"
    ]

    cluster_profile = cluster_df.groupby("Cluster")[cluster_features].mean()
    st.dataframe(cluster_profile, use_container_width=True)

    st.markdown("### Countries by Cluster")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(cluster_df["Cluster"].unique())
    )

    st.dataframe(
        cluster_df[cluster_df["Cluster"] == selected_cluster][["Country", "Cluster"]],
        use_container_width=True
    )