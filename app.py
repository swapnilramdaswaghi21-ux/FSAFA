import streamlit as st
import pandas as pd
from credit_model import load_model, FEATURES

st.set_page_config(
    page_title="QR-XFE Credit Scanner",
    page_icon="🏦",
    layout="wide"
)

# ===================== HEADER =====================
st.markdown("<h1 style='text-align:center'>🏦 QR-XFE Corporate Credit Scanner</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#64748b;font-size:1.2rem'>Live Loan Fraud Detection | ₹100Cr+ Exposure</p>", unsafe_allow_html=True)
st.markdown("---")

model, accuracy, credit_data = load_model()

# ===================== KPIs =====================
c1,c2,c3,c4 = st.columns(4)
c1.metric("Loans Screened", "2,500")
c2.metric("Rejected", "18%")
c3.metric("Model Accuracy", f"{accuracy:.1%}")
c4.metric("Basel III", "✅")

# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs(["🔍 Loan Scan","🚫 Reject List","📊 Portfolio"])

# ===================== TAB 1 =====================
with tab1:
    st.subheader("New Loan Application")

    col1, col2 = st.columns(2)
    with col1:
        beneish = st.number_input("Beneish M-Score", -4.0, 1.0, -2.4)
        altman = st.number_input("Altman Z-Score", 0.0, 6.0, 2.6)
        sloan = st.number_input("Sloan Ratio %", -0.1, 0.1, 0.01) / 100
        dsr = st.number_input("DSR", 0.5, 5.0, 1.4)

    with col2:
        current_ratio = st.number_input("Current Ratio", 0.5, 4.0, 1.3)
        wc_days = st.number_input("Working Capital Days", 30, 200, 90)
        auditor = st.checkbox("Auditor Qualification")

    if st.button("🚀 ASSESS CREDIT RISK", use_container_width=True):
        app = pd.DataFrame([[
            beneish, altman, sloan, 6,
            dsr, current_ratio, wc_days, int(auditor)
        ]], columns=FEATURES)

        risk = model.predict_proba(app)[0,1]

        if risk > 0.75:
            st.error(f"🚫 LOAN REJECTED | Fraud Risk {risk:.0%}")
        elif risk > 0.45:
            st.warning(f"⚠️ CONDITIONAL APPROVAL | Risk {risk:.0%}")
        else:
            st.success(f"✅ APPROVED | Risk {risk:.0%}")

# ===================== TAB 2 =====================
with tab2:
    st.subheader("High Risk Loan Applications")
    st.dataframe(
        credit_data[credit_data['fraud_flag']==1]
        .nlargest(15,'Beneish_M')
        [['company','industry','loan_amount_cr','DSR']]
        .round(2),
        use_container_width=True
    )

# ===================== TAB 3 =====================
with tab3:
    portfolio = pd.DataFrame({
        "Company":["ABC Steel","XYZ Realty","LMN Pharma"],
        "Exposure ₹Cr":[250,180,320],
        "Risk":[0.12,0.78,0.08],
        "Status":["Normal","Watchlist","Normal"]
    })
    st.dataframe(portfolio, use_container_width=True)

st.markdown("---")
st.markdown("<center>© 2025 QR-XFE | Bank Credit Risk Platform</center>", unsafe_allow_html=True)

