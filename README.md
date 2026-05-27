# Global Economic Prediction Dashboard

A machine learning-powered economic analytics dashboard built using Python and Streamlit to analyze global economic indicators, forecast inflation trends, detect recession risks, and visualize country-level economic performance.

## Live Demo

[https://global-economic-prediction-dashboard-qldrufuvmj3v8khn6mjged.streamlit.app/]

## GitHub Repository

[https://github.com/nesaugust/global-economic-prediction-dashboard]

---

# Project Overview

This project combines:

* Data Analytics
* Machine Learning
* Forecasting
* Economic Risk Analysis
* Interactive Visualization

to create a complete global economic intelligence dashboard.

The dashboard helps users explore worldwide economic trends, compare countries, detect high-risk economies, and generate AI-based economic predictions.

---

# Dashboard Features

## 1. Global Overview

* Global GDP trend
* Inflation analysis
* Unemployment trend
* KPI summary cards

## 2. Country Comparison

* Compare multiple countries
* Economic rankings
* Economic score analysis

## 3. Trade Analytics

* Export and import analysis
* Trade balance visualization
* International trade comparison

## 4. Financial Stability

* Foreign reserve monitoring
* Exchange rate analysis
* Stock market indicators

## 5. AI Prediction

* Inflation prediction model
* Recession probability prediction
* Economic forecasting

## 6. Global Risk Intelligence

* Top risky countries
* Anomaly detection
* Crisis alert system

## 7. Country Clustering

* Economic segmentation
* PCA visualization
* K-Means clustering

---

# Technologies Used

## Programming Language

* Python

## Data Processing

* Pandas
* NumPy

## Visualization

* Plotly
* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Dashboard Framework

* Streamlit

---

# Project Structure

```text id="3zw1sa"
global-economic-prediction-dashboard/
│
├── EDA_FeatureEngineering/
│   ├── global_economy_clean_dataset.csv
│   ├── global_economy_dataset.csv
│   └── mergedata.ipynb
│
├── Prediction_Data/
│   ├── anomaly_detection_data.csv
│   ├── country_cluster_data.csv
│   ├── economic_score_data.csv
│   ├── final_global_economy_data.csv
│   ├── recession_probability_data.csv
│   └── risk_score_data.csv
│
├── Raw Data/
│   ├── CPI Price datasets
│   ├── GDP datasets
│   ├── Exchange Rate datasets
│   ├── Trade datasets
│   ├── Unemployment datasets
│   └── Financial datasets
│
├── models/
│   ├── anomaly_model.pkl
│   ├── inflation_model.pkl
│   ├── pca_model.pkl
│   ├── recession_model.pkl
│   └── scaler.pkl
│
├── app.py
├── data_modelling.ipynb
├── processed_global_economy_data.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Machine Learning Models

## Inflation Prediction Model

Predicts future inflation trends using historical economic indicators.

## Recession Prediction Model

Estimates recession probability using macroeconomic features.

## Anomaly Detection Model

Detects countries with unusual economic conditions and potential crisis signals.

## Country Clustering

Groups countries into economic segments using:

* K-Means Clustering
* PCA Dimensionality Reduction

---

# Project Workflow

## Step 1 — Raw Data Collection

Collected multiple global economic datasets including:

* GDP
* Inflation
* Unemployment
* Exports & Imports
* Foreign Reserves
* Exchange Rates
* Stock Market Data

## Step 2 — Data Cleaning & Feature Engineering

Performed:

* Missing value handling
* Data merging
* Feature engineering
* Country-year aggregation
* Economic score generation

## Step 3 — Exploratory Data Analysis (EDA)

Analyzed:

* Global economic trends
* Country comparison
* Correlation analysis
* Risk patterns

## Step 4 — Machine Learning Modelling

Built:

* Inflation prediction model
* Recession classifier
* Economic clustering model
* Anomaly detection system

## Step 5 — Dashboard Development

Created an interactive Streamlit dashboard with:

* Dynamic filters
* Interactive charts
* Forecasting visualizations
* Risk analytics

## Step 6 — Deployment

Deployed the project using Streamlit Cloud and GitHub integration.

---

# Installation

Clone the repository:

```bash id="xw33bk"
git clone https://github.com/nesaugust/global-economic-prediction-dashboard.git
```

Move into the project folder:

```bash id="p7xiy9"
cd global-economic-prediction-dashboard
```

Install dependencies:

```bash id="0v5wws"
pip install -r requirements.txt
```

Run the Streamlit app:

```bash id="dgtjif"
streamlit run app.py
```

---

# Future Improvements

* Real-time economic API integration
* Deep learning forecasting models
* Interactive world heatmaps
* Advanced economic simulations
* Cloud database integration

---

# Author

Agnes Jeni Makay
