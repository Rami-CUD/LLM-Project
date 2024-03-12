import os
import getpass
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

if "GOOGLE_API_KEY" not in os.environ:
    API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')

st.title = "Testing"

model = GoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY)

with st.form("my_form"):
    text = st.text_area("Enter Text")
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.info(model.invoke(text))