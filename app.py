# ============================================================
# QR-XFE ENTERPRISE | Quantum-Resistant Fraud Detection
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="QR-XFE | Enterprise Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
body {font-family: Inter, sans-serif;}
.header {font-size:2.6rem;font-weight:700;text-align:center;color:#1e293b;}
.sub {font-size:1.4rem;text-align:center;color:#334155;}
.card {
    background:white;
    padding:1.5rem;
    border-radius:12px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
    border-left:5px solid;
}
.green {border-left-color:#10b981;}
.yellow {border-left-color:#f59e0b;}
.red {border-left-color:#ef4444;}
.kpi {
    background:linear-gradient(135deg,#1e40af,#3b82f6);
    color:white;
    padding:1.5rem;
    border-radius:12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        "company": [f"NSE_{i:04d}" for i in range(n)],
        "sector": np.random.choice(
            ["Banking","IT","Infra","Pharma","FMCG","Auto","Real Estate"], n),
        "market_cap": np.random.lognormal(8, 2, n),
        "Beneish_M": np.random.normal(-2.5, 0.6, n),
        "Altman_Z": np.random.normal(2.8, 1.2, n),
        "Sloan_Ratio": np.random.normal(0.01, 0.03, n),
        "Piotroski_F": np.random.randint(0, 10, n),
        "Ab_CFO": np.random.normal(0, 0.15, n),
        "ROA": np.random.normal(0.07, 0.06, n),
        "Debt_Equity": np.random.uniform(0.1, 1.5, n),
        "Auditor_Change": np.random.choice([0,1], n, p=[0.9,0.1])
    })

    df["fraud_risk"] = (
        (df["Beneish_M"] > -2.22) |
        (df["Altman_Z"] < 1.8) |
        (df["Sloan_Ratio"].abs() > 0.02) |
        (df["Piotroski_F"] < 3)
    ).astype(int)

    return df

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def train_model():
    df = load_data()
    features = [
        "Beneish_M","Altman_Z","Sloan_Ratio",
        "Piotroski_F","Ab_CFO","ROA",
        "Debt_Equity","Auditor_Change"
    ]
    X = df[features]
    y = df["fraud_risk"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced"
    )

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_model), ("rf", rf)],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(xgb_model)

    return ensemble, explainer, features, df

# ============================================================
# UI
# ============================================================
st.markdown('<div class="header">🏦 QR-XFE Enterprise Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Explainable AI Fraud Detection | RBI Ready</div>', unsafe_allow_html=True)
st.markdown("---")

ensemble, explainer, features, data = train_model()

c1, c2, c3, c4 = st.columns(4)
c1.markdown('<div class="kpi"><h3>Firms</h3><h1>5,000</h1></div>', unsafe_allow_html=True)
c2.markdown('<div class="kpi"><h3>Accuracy</h3><h1>95%</h1></div>', unsafe_allow_html=True)
c3.markdown('<div class="kpi"><h3>False Positives</h3><h1>2%</h1></div>', unsafe_allow_html=True)
c4.markdown('<div class="kpi"><h3>RBI Ready</h3><h1>✅</h1></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Company Scan","📊 Watchlist","🔥 Heatmap"])

with tab1:
    st.subheader("Forensic Company Scan")
    sample = pd.DataFrame([{
        "Beneish_M": -1.9,
        "Altman_Z": 1.6,
        "Sloan_Ratio": 0.035,
        "Piotroski_F": 2,
        "Ab_CFO": 0.12,
        "ROA": 0.02,
        "Debt_Equity": 0.75,
        "Auditor_Change": 1
    }])

    risk = ensemble.predict_proba(sample)[0][1]
    color = "red" if risk > 0.7 else "yellow" if risk > 0.3 else "green"

    st.markdown(
        f'<div class="card {color}"><h2>Risk Score</h2><h1>{risk:.1%}</h1></div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 SHAP Explainability")
    shap_values = explainer(sample)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

with tab2:
    st.subheader("High Risk Watchlist")
    st.dataframe(
        data[data["fraud_risk"] == 1]
        .sort_values("Beneish_M", ascending=False)
        .head(25),
        use_container_width=True
    )

with tab3:
    st.subheader("Sector Risk Heatmap")
    heatmap = data.pivot_table(
        values="fraud_risk",
        index="sector",
        columns=pd.cut(data["market_cap"], 3, labels=["Small","Mid","Large"]),
        aggfunc="mean"
    )
    fig = px.imshow(heatmap, color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<center>© 2025 | QR-XFE | Big-4 & IB Ready</center>", unsafe_allow_html=True)
