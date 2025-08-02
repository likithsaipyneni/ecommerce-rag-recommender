import streamlit as st

st.set_page_config(page_title="E-commerce RAG Recommender", layout="wide")

st.title("🛍️ E-commerce RAG Product Recommender")
st.markdown("This is a placeholder app. Please integrate the full pipeline here.")

# Example input
query = st.text_input("Enter your query", "")

if query:
    st.success(f"You searched for: {query}")
