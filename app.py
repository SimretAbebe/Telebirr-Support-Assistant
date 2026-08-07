import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from rag_pipeline import ask

st.set_page_config(page_title="Telebirr Support Assistant")

st.title("Telebirr Support Assistant")
st.write("Ask a question about sending money, cash-out, or failed transactions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(user_question)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})