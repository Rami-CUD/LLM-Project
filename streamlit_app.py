import os
import getpass
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

if "GOOGLE_API_KEY" not in os.environ:
    # os.environ["GOOGLE_API_KEY"] = getpass.getpass("API Key: ")
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCILLp4kYKQKVW8BWmXE2Hh4fomiZwXdfU"

st.title = "Testing"

model = GoogleGenerativeAI(model="gemini-pro")

with st.form("my_form"):
    text = st.text_area("Enter Text")
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.info(model.invoke(text))