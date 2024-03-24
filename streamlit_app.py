# importing necessary files 
# 'import as' imports a module and gives it an alias 
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from pypdf import PdfReader
from typing import IO, Union
from io import StringIO

#Sets title and icon to be display in browser tabs
st.set_page_config(page_title="ChatGemini", page_icon="🤖")

# format for a CSS style 
css = """
    <style>
    .chatgemini-heading {
        font-size: 50px;
        font-weight: bold;
        color: #fcad03;
        text-align: center;
    }
    </style>
"""

# applying CSS style
st.markdown(css, unsafe_allow_html=True)

# render the heading
st.markdown('<h1 class="chatgemini-heading">ChatGemini</h1>', unsafe_allow_html=True)
MAX_PROMPT_CHAR_COUNT = 30500

# when the user first enters the website, keys are first initilized to save the user state 
if "stack" not in st.session_state:
    st.session_state.stack = []
if "pdf_state_changed" not in st.session_state:
    st.session_state.pdf_state_changed = True
if "file_content" not in st.session_state:
    st.session_state.file_content = ""

# Function that goes through the history list and display the chat history
def display_chat_history(chat:list[tuple[2]]):
    for prompt, response in chat:
        messages.chat_message("User").write(prompt)
        messages.chat_message("AI").write(response)

# Uses PDFreader to read each page in PDF and add it to one big String
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

# stores a pdf file in the user's state and displays warning if file is too big
def change_pdf_state(PDF):
    st.session_state.file_content = get_pdf_content(PDF)
    if len(st.session_state.file_content) > MAX_PROMPT_CHAR_COUNT:
        st.warning("PDF length is greater than supported length. Prompt might not work...")

# Will get called when the user changes the state of the st.file_uploader object
def on_change_func():
    st.session_state.pdf_state_changed = True

# A session key and title is hard coded for program simplicity and to avoid user confusion
st.session_state.key = "AIzaSyCILLp4kYKQKVW8BWmXE2Hh4fomiZwXdfU"
st.title = "ChatGemini"

# Gemini initilized with top k, top p, and temperature parameters 
model = GoogleGenerativeAI(model="gemini-pro", google_api_key=st.session_state.key, max_retries=6, top_k=10, top_p=0.9, temperature=0.65)

# Two filler columns are placed between col1 and col2 for spacing
col1, *_, col2 = st.columns(4, gap="large")
with col1:
    with st.popover("Upload a File"):
        PDFFile = st.file_uploader(label="dds", type=".pdf", label_visibility="collapsed", on_change=on_change_func)
        if st.session_state.pdf_state_changed:
            if PDFFile:
                change_pdf_state(PDFFile)
            else:
                st.session_state.file_content = ""    
            st.session_state.pdf_state_changed = False
with col2:
    if st.button("Clear History"):
        st.session_state.stack = []

# Container that holds the messages 
messages = st.container()
# Chat input UI element
chat = st.chat_input("Enter Text")

# Constantly display the chat history
display_chat_history(st.session_state.stack)

#If user inputs something in the chat_input, display his message, get and display the AI response, and add both to the history
if chat:
    messages.chat_message("User").write(chat)
    try:
        if st.session_state.file_content:
            prompt = f"Context: {st.session_state.file_content} Prompt: {chat}"
        else:
            prompt = chat        
        current_response = model.invoke(prompt)
        if not current_response:
            raise Exception("Empty Response")
            
    # two exceptions may occur
    except IndexError:
        # this error is when the response is restricted (based on Google's terms and conditions)
        current_response = ":red[Sorry, I can not respond to this prompt...]"
    except Exception:
        # general exceptions
        current_response = ":red[An error has occured...]"

    # displaying Gemini's responses
    messages.chat_message("AI").write(current_response)

    # adding Gemini's response to the history stack
    st.session_state.stack.append((chat, current_response))
