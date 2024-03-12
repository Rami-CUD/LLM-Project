import os
import getpass
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

if "stack" not in st.session_state or "key" not in st.session_state:
    st.session_state.stack = []
    
st.session_state.key = "AIzaSyCILLp4kYKQKVW8BWmXE2Hh4fomiZwXdfU"
st.title = "Testing"

model = GoogleGenerativeAI(model="gemini-pro", google_api_key=st.session_state.key)

chat = st.chat_input("Enter Text")
messages = st.container()
if chat:
    for prompt, response in st.session_state.stack:
        messages.chat_message("User").write(prompt)
        messages.chat_message("AI").write(response)
    messages.chat_message("User").write(chat)
    try:        
        current_response = model.invoke(chat)
    except IndexError:
        current_response = "Sorry, I can not respond to this prompt..."
    messages.chat_message("AI").write(current_response)
    st.session_state.stack.append((chat, current_response))