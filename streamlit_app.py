import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from pypdf import PdfReader
<<<<<<< HEAD
=======
from typing import IO, Union
from io import StringIO
>>>>>>> 62cf4f381ce05cfef8792de85e7a803f1d66319a

MAX_PROMPT_CHAR_COUNT = 30500
if "stack" not in st.session_state:
    st.session_state.stack = []
if "pdf_state_changed" not in st.session_state:
    st.session_state.pdf_state_changed = True
if "file_content" not in st.session_state:
    st.session_state.file_content = ""

def display_chat_history(chat:list[tuple[2]]):
    for prompt, response in chat:
        messages.chat_message("User").write(prompt)
        messages.chat_message("AI").write(response)

def get_pdf_content(file:Union[IO, str]):
    reader = PdfReader(file)
    string_buffer = StringIO()
    pdf_length = len(reader.pages)
    progress_bar = st.progress(0.0)
    for i, page in enumerate(reader.pages):
        string_buffer.write(page.extract_text())
        progress_bar.progress(i/pdf_length)
    progress_bar.empty()
    return string_buffer.getvalue()

def change_pdf_state(PDF):
    st.session_state.file_content = get_pdf_content(PDF)
    if len(st.session_state.file_content) > MAX_PROMPT_CHAR_COUNT:
        st.warning("PDF length is greater than supported length. Prompt might not work...")
    
def on_change_func():
    st.session_state.pdf_state_changed = True


st.session_state.key = "AIzaSyCILLp4kYKQKVW8BWmXE2Hh4fomiZwXdfU"
st.title = "Testing"
model = GoogleGenerativeAI(model="gemini-pro", google_api_key=st.session_state.key)
with st.popover("Upload a File"):
    PDFFile = st.file_uploader(label="dds", type=".pdf", label_visibility="collapsed", on_change=on_change_func)
    if st.session_state.pdf_state_changed:
        if PDFFile:
            change_pdf_state(PDFFile)
        else:
            st.session_state.file_content = ""    
        
        st.session_state.pdf_state_changed = False

messages = st.container()
chat = st.chat_input("Enter Text")
display_chat_history(st.session_state.stack)
if chat:
    
    #Start displaying new message
    messages.chat_message("User").write(chat)
    try:
        if st.session_state.file_content:
            prompt = f"Context: {st.session_state.file_content} Prompt: {chat}"
        else:
            prompt = chat        
        current_response = model.invoke(prompt)
    except IndexError:
        current_response = "Sorry, I can not respond to this prompt..."
    except Exception:
        current_response = ":red[An error has occured...]"
    
    messages.chat_message("AI").write(current_response)
    st.session_state.stack.append((chat, current_response))

