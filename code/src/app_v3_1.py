import streamlit as st
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langgraph.graph import StateGraph
from langgraph.graph.state import State

# Load policy document into vector database
loader = PyPDFLoader("Capital_Assessments_and_Stress_Testing.pdf")
documents = loader.load()
embedding_model = OpenAIEmbeddings()
vector_db = FAISS.from_documents(documents, embedding_model)

# Load LLaMA Model (Groq LLaMA)
llm = LlamaCpp(model_path="path/to/groq-llama-model.gguf")

def retrieve_policy(risk_type):
    docs = vector_db.similarity_search(risk_type, k=1)
    if docs:
        summary = llm(docs[0].page_content[:3000])  # Summarizing policy content
        return summary
    return "No relevant policy found."

# Define workflow state
class RiskState(State):
    risk_type: str = ""
    policy: str = ""
    update_needed: bool = False
    updated_policy: str = ""

# Define LangGraph workflow
workflow = StateGraph(RiskState)

def fetch_policy(state: RiskState) -> RiskState:
    state.policy = retrieve_policy(state.risk_type)
    return state

workflow.add_node("fetch_policy", fetch_policy)

def ask_policy_update(state: RiskState) -> RiskState:
    st.write(f"### Policy for '{state.risk_type}':")
    st.write(state.policy)
    state.update_needed = st.radio("Do you want to update this policy?", ["No", "Yes"]) == "Yes"
    return state

workflow.add_node("ask_policy_update", ask_policy_update)

def update_policy(state: RiskState) -> RiskState:
    if state.update_needed:
        state.updated_policy = st.text_area("Enter the updated policy:")
        if st.button("Update Policy"):
            st.success("✅ Policy updated successfully.")
    return state

workflow.add_node("update_policy", update_policy)
workflow.set_entry_point("fetch_policy")
workflow.add_edge("fetch_policy", "ask_policy_update")
workflow.add_edge("ask_policy_update", "update_policy")
executor = workflow.compile()

# Streamlit UI
st.title("Transaction Risk Analysis & Policy Update")

# User inputs
transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
country = st.selectbox("Country", ["USA", "UK", "Germany", "Japan", "France"])
currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
risk_type = st.selectbox("Select Transaction Risk Type", [
    "High-Risk Country Transaction", "Currency Mismatch", "Round-Number Transaction",
    "Cross-Border Limit Violation", "Future Transaction Date"
])

if st.button("Analyze Risk"):
    initial_state = RiskState(risk_type=risk_type)
    final_state = executor.invoke(initial_state)
