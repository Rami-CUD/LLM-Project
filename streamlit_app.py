import streamlit as st
from langchain_google_genai import GoogleGenerativeAI

if "stack" not in st.session_state or "key" not in st.session_state:
    st.session_state.stack = []
    
st.session_state.key = "AIzaSyCILLp4kYKQKVW8BWmXE2Hh4fomiZwXdfU"
st.title = "Testing"
with st.echo():
    st.write(st.__version__)
model = GoogleGenerativeAI(model="gemini-pro", google_api_key=st.session_state.key)
# with st.popover("Upload a File"):
#     st.file_uploader(label="dds", type=".pdf", label_visibility="collapsed")
messages = st.container()
chat = st.chat_input("Enter Text")
if chat:
    for prompt, response in st.session_state.stack:
        messages.chat_message("User").write(prompt)
        messages.chat_message("AI").write(response)
    messages.chat_message("User").write(chat)
    try:        
        current_response = model.invoke(chat)
    except IndexError:
        current_response = "Sorry, I can not respond to this prompt..."
    except Exception:
        current_response = ":red[An error has occured...]"
    
    messages.chat_message("AI").write(current_response)
    st.session_state.stack.append((chat, current_response))