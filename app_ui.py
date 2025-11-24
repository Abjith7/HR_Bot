# app_ui.py
import streamlit as st
from app_backend import answer_query

st.set_page_config(layout="wide", page_title="RAG Chatbot")
st.title("RAG Chatbot — local (Chroma) + HF Inference")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about your documents:")

if st.button("Ask") and query.strip():
    st.session_state.history.append({"user": query, "bot": "..."})
    # call backend
    try:
        answer = answer_query(query)
    except Exception as e:
        answer = f"Error: {e}"
    st.session_state.history[-1]["bot"] = answer

for turn in reversed(st.session_state.history):
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Bot:** {turn['bot']}")
    st.divider()
