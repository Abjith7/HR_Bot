import streamlit as st
from app_backend import answer_query
import time

st.set_page_config(layout="wide", page_title="🧞 Genie — Conversational RAG")

# -------- SESSION STATE --------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# -------- CSS --------
st.markdown("""
<style>
.chat-container {
    width: 100%;
    max-width: 900px;
    margin: auto;
    padding-bottom: 120px;
}

.user-msg {
    background-color: #1e1e1e;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 10px 0 4px auto;
    width: fit-content;
    max-width: 75%;
    color: white;
}

.genie-msg {
    background-color: #2b2b2b;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 4px auto 6px 0;
    width: fit-content;
    max-width: 75%;
    color: white;
}

.meta-box {
    font-size: 13px;
    color: #cfcfcf;
    margin: 2px 0 12px 12px;
}

.input-box {
    position: fixed;
    bottom: 20px;
    width: 90%;
    max-width: 900px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------- TITLE --------
st.title("🧞 Genie — Your Conversational RAG Assistant")

# -------- SEND FUNCTION --------
def submit():
    user_query = st.session_state.user_input.strip()
    if not user_query:
        return

    st.session_state.user_input = ""

    # Store user message
    st.session_state.chat.append({
        "role": "user",
        "content": user_query
    })

    with st.spinner("Genie is thinking..."):
        result = answer_query(user_query)

    st.session_state.chat.append({
        "role": "genie",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "latency": result.get("latency", {})
    })


# -------- CHAT RENDER --------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-msg'><b>You:</b> {msg['content']}</div>",
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"<div class='genie-msg'><b>Genie:</b> {msg['content']}</div>",
            unsafe_allow_html=True
        )

        # -------- SOURCES --------
        if msg.get("sources"):
            src_block = "<br>".join(msg["sources"])
            st.markdown(
                f"<div class='meta-box'><b>Sources:</b><br>{src_block}</div>",
                unsafe_allow_html=True
            )

        # -------- LATENCY --------
        if msg.get("latency"):
            lat = msg["latency"]
            st.markdown(
                f"<div class='meta-box'><b>Latency:</b> "
                f"Retrieval {lat['retrieval_time']}s | "
                f"Generation {lat['generation_time']}s | "
                f"Total {lat['total_time']}s</div>",
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)


# -------- INPUT BAR --------
st.markdown("<div class='input-box'>", unsafe_allow_html=True)

col1, col2 = st.columns([8, 1])
with col1:
    st.text_input(
        "",
        key="user_input",
        placeholder="Ask Genie something about your documents...",
        on_change=submit
    )

with col2:
    if st.button("Send"):
        submit()

st.markdown("</div>", unsafe_allow_html=True)
