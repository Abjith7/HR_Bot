import streamlit as st
from app_backend import answer_query
import time

st.set_page_config(layout="wide", page_title="RAG Chatbot")

# --- Initialize session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- CSS for fixed prompt box and chat ---
st.markdown("""
<style>
body { background-color: #121212; color: white; }
#chat-container { padding: 10px 10px 80px 10px; } /* bottom padding for fixed input */
.chat-user { background-color: #1e1e1e; color: #fff; padding: 10px 15px; border-radius: 15px; width: fit-content; max-width: 70%; margin-left: auto; margin-bottom: 5px; word-wrap: break-word; }
.chat-bot { background-color: #2b2b2b; color: #fff; padding: 10px 15px; border-radius: 15px; width: fit-content; max-width: 70%; margin-right: auto; margin-bottom: 10px; word-wrap: break-word; }

.input-container { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); width: 95%; max-width: 700px; display: flex; z-index: 1000; }
.input-container input[type="text"] { flex: 1; color: #fff; background-color: #1a1a1a; border: none; outline: none; padding: 12px 15px; border-radius: 15px 0 0 15px; font-size: 16px; }
.input-container button { border: none; padding: 12px 20px; border-radius: 0 15px 15px 0; background-color: #4caf50; color: white; font-size: 16px; cursor: pointer; }
.input-container button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

# --- Submit function ---
def submit():
    user_query = st.session_state.user_input.strip()
    if not user_query:
        return
    st.session_state.history.append({"user": user_query, "bot": ""})
    st.session_state.user_input = ""  # clear input

    bot_placeholder = st.empty()
    try:
        reply = answer_query(user_query)
    except Exception as e:
        reply = f"Error: {e}"

    # Stream reply character by character
    full_reply = ""
    for char in reply:
        full_reply += char
        bot_placeholder.markdown(f"<div class='chat-bot'><b>Bot:</b> {full_reply}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

    st.session_state.history[-1]["bot"] = reply

# --- Chat container ---
st.markdown('<div id="chat-container">', unsafe_allow_html=True)
for turn in st.session_state.history:
    st.markdown(f"<div class='chat-user'><b>You:</b> {turn['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bot'><b>Bot:</b> {turn['bot']}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Fixed input + Send button ---
input_col, button_col = st.columns([8, 1], gap="small")
with input_col:
    st.text_input("", key="user_input", placeholder="Type your message...", on_change=submit)
with button_col:
    if st.button("Send"):
        submit()
