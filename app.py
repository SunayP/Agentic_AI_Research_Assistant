# app.py

import streamlit as st
from chatbot_core import generate_response

# Page configuration
st.set_page_config(page_title="Multi-Agent Chatbot", layout="wide")

# Title
st.title("🤖 Multi-Agent AI Chatbot")

# Session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything...", key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    with st.spinner("Thinking..."):
        response = generate_response(user_input)
    
    # Save to history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
st.markdown("---")
for speaker, message in reversed(st.session_state.chat_history):
    if speaker == "You":
        st.markdown(f"<div style='text-align: right;'><strong>{speaker}:</strong> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left;'><strong>{speaker}:</strong> {message}</div>", unsafe_allow_html=True)