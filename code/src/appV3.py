import streamlit as st
import pandas as pd
import numpy as np
import joblib
import langgraph.graph as lg
from langchain.chat_models import ChatGroq
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load ML Model and Encoders
model = joblib.load('risk_scoring_model.pkl')
le_country = joblib.load('country_encoder.pkl')
le_currency = joblib.load('currency_encoder.pkl')

# Load vector database
vector_store = FAISS.load_local("policy_vector_db", OpenAIEmbeddings())

# Load policy document
policies = pd.read_csv("capital_assessment_policies.csv")

# Initialize Groq LLaMA model
llm = ChatGroq(model="llama3-70b")

def analyze_risk(transaction):
    """Calculate risk score & type"""
    input_data = np.array([[transaction["amount"], le_country.transform([transaction["country"]])[0],
                            le_currency.transform([transaction["currency"]])[0], transaction["balance"],
                            transaction["overdraft"]]])
    risk_score = model.predict(input_data)[0][0]
    risk_category = "Low" if risk_score <= 3 else "Medium" if risk_score <= 6 else "High"
    return risk_score, risk_category

def retrieve_policy(risk_type):
    """Fetch relevant policy from RAG system"""
    query = f"Policy for {risk_type} risk in Capital Assessments"
    docs = vector_store.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No specific policy found."

def update_policy(policy_id, new_text):
    """Update policy in database"""
    policies.loc[policies['policy_id'] == policy_id, 'policy_text'] = new_text
    policies.to_csv("capital_assessment_policies.csv", index=False)
    return "Policy updated successfully."

# Streamlit UI
st.title("🔍 Transaction Risk Analysis & Policy Compliance")

# User Inputs
amount = st.number_input("Transaction Amount ($)", min_value=1.0, step=0.01)
country = st.selectbox("Country", policies["country"].unique())
currency = st.selectbox("Currency", policies["currency"].unique())
balance = st.number_input("Account Balance ($)", min_value=-10000.0, step=0.01)
overdraft = st.checkbox("Overdraft Account")

if st.button("Analyze Transaction"):
    # Risk Analysis
    transaction = {"amount": amount, "country": country, "currency": currency, "balance": balance, "overdraft": overdraft}
    risk_score, risk_category = analyze_risk(transaction)
    
    # Fetch Policy
    policy_text = retrieve_policy(risk_category)
    
    # Display Results
    st.subheader("Risk Analysis Results")
    st.write(f"**Risk Score:** {risk_score:.2f}")
    st.write(f"**Risk Category:** {risk_category}")
    st.subheader("Relevant Policy")
    st.write(policy_text)
    
    # Allow Policy Update
    if st.checkbox("Update Policy?"):
        new_policy = st.text_area("Modify Policy Text", value=policy_text)
        if st.button("Save Updated Policy"):
            update_policy(policy_id=risk_category, new_text=new_policy)
            st.success("Policy updated successfully!")
