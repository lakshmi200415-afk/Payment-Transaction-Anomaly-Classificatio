import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Payment Anomaly Detection", page_icon="💳")
st.title("💳 Payment Transaction Anomaly Detection")
st.write("AI model to detect fraudulent / anomalous transactions")

# Load model
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Model loaded successfully!")
except:
    st.error("Model not found! Please run train.py first")
    st.stop()

st.divider()
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Amount (₹)", min_value=10, max_value=100000, value=5000)
    time = st.slider("Transaction Hour (0-23)", 0, 23, 14)
    frequency = st.slider("Transactions Today", 1, 20, 3)
with col2:
    merchant_risk = st.selectbox("Merchant Risk", [0, 1], format_func=lambda x: "Low" if x==0 else "High")
    account_age = st.number_input("Account Age (days)", min_value=1, max_value=2000, value=300)

if st.button("🔍 Check for Anomaly", type="primary"):
    features = np.array([[amount, time, frequency, merchant_risk, account_age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]

    if prediction == 1:
        st.error(f"🚨 ANOMALY DETECTED! Fraud Risk: {prob[1]*100:.1f}%")
        st.write("This transaction looks suspicious.")
    else:
        st.success(f"✅ NORMAL TRANSACTION. Safe: {prob[0]*100:.1f}%")
        st.write("This transaction looks normal.")

st.divider()
st.caption("Project: Payment Anomaly Detection | Model: Logistic Regression | Accuracy: 86%")