import streamlit as st
import numpy as np
import joblib

# Load model & encoders
model = joblib.load('risk_scoring_model.pkl')
le_country = joblib.load('country_encoder.pkl')
le_currency = joblib.load('currency_encoder.pkl')

# Streamlit UI
st.title("🚨 Transaction Risk Analyzer 🚨")
st.markdown("Enter transaction details to analyze **risk score (1-10) and risk type**.")

# Input Fields
transaction_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=500000.0, value=1000.0)
reported_amount = st.number_input("Reported Amount ($)", min_value=1.0, max_value=500000.0, value=1000.0)
account_balance = st.number_input("Account Balance ($)", min_value=-5000.0, max_value=200000.0, value=5000.0)
overdraft_flag = st.radio("Overdraft Account?", ["Yes", "No"])
currency = st.selectbox("Currency", le_currency.classes_)
country = st.selectbox("Country", le_country.classes_)

# Convert inputs
overdraft_flag = True if overdraft_flag == "Yes" else False
encoded_country = le_country.transform([country])[0]
encoded_currency = le_currency.transform([currency])[0]

# Predict risk
if st.button("Analyze Risk"):
    input_data = np.array([[transaction_amount, encoded_country, encoded_currency, account_balance, overdraft_flag]])
    risk_score = model.predict(input_data)[0][0]  # Get risk score

    # Categorize risk based on score
    if risk_score <= 3:
        risk_category = "Low Risk ✅"
    elif 4 <= risk_score <= 6:
        risk_category = "Medium Risk ⚠️"
    else:
        risk_category = "High Risk 🚨"

    st.subheader(f"Risk Score: **{round(risk_score, 1)}/10** ({risk_category})")

    # Additional explanations
    if risk_score > 6:
        st.warning("🚨 This transaction requires compliance review.")
    elif risk_score > 4:
        st.info("⚠️ Additional validation might be needed.")

    st.markdown("🔍 **Automated AI-powered risk assessment.**")
