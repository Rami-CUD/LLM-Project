# importing necessary files 
# 'import as' imports a module and gives it an alias 
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from pypdf import PdfReader
from typing import IO, Union
from io import StringIO, BytesIO
from BytesPDFLoader import BytesIOPyPDFLoader



# Loads enviroment variables stores in .env file
load_dotenv()

# Sets title and icon to be display in browser tabs
st.set_page_config(page_title="ChatGemini", page_icon="🤖")

# CSS style for heading
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

# Applying CSS style
st.markdown(css, unsafe_allow_html=True)

# Render the heading
st.markdown('<h1 class="chatgemini-heading">ChatGemini</h1>', unsafe_allow_html=True)

# Initializing session state variables (keys) 
if "history_list" not in st.session_state:
    st.session_state.history_list = []
if "pdf_state_changed" not in st.session_state:
    st.session_state.pdf_state_changed = True
if "pdf_vector_store" not in st.session_state:
    st.session_state.pdf_vector_store = None

# Function that goes through the history list and display the chat history
def display_chat_history(history:list[BaseMessage]):
    user_message = True
    for message in history:
        message_sender = "User" if user_message else "AI"
        messages.chat_message(message_sender).write(message.content)
        user_message = not user_message


# Will get called when the user changes the state of the st.file_uploader object
def on_change_func():
    st.session_state.pdf_state_changed = True


# Initalize Langchain [Documents] from uploaded file and split them up into chunks 
def get_docs(file: BytesIO):
    loader = BytesIOPyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    split_documents = text_splitter.split_documents(documents)
    return split_documents

# Vector Store database allows efficient retriaval of relevant information from the document
def create_vector_store(docs):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embedding=embedding)
    return vector_store



# Create a langchain chain depending on whether the user provided context or not (context from the PDF document)
def create_chain(with_context: bool, model):
    messages_list = [
        ("system", "Answer the user's questions in a friendly and respectful manner, and make sure to NOT prefix your responses with AI: or System:"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ]
    if not with_context:
        prompt = ChatPromptTemplate.from_messages(messages_list)
        chain = prompt | model
        return chain
    
    messages_list[0] = ("system", "Without prefixing your responses with AI: or System: , answer the user's questions based on the context: {context}")
    prompt = ChatPromptTemplate.from_messages(messages_list)
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    vectorStore = st.session_state.pdf_vector_store
    retriever = vectorStore.as_retriever(search_kwargs= {"k": 20})
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return retrieval_chain


# Use stream method of chains to get AI response in chunks and display them as they come to simulate the response
# being formulated in real time similar to ChatGPT.
# The empty output_container allows the text to be replaced every loop instead of being appended to the chat message
def get_response_while_showing_stream(chain, prompt, chat_history, container) -> str:
    with container.chat_message("AI"):
        text = ""
        output_container = st.empty()
        try:
            with st.spinner("Generating..."):
                for chunk in chain.stream({"input": prompt, "chat_history": chat_history}):
                        if isinstance(chunk, dict) and "answer" in chunk:
                            text += chunk["answer"]
                        elif isinstance(chunk, str):
                            text += chunk
                        output_container.write(text)
        # IndexError usually occurs when the Gemini's response is restricted
        # Exception handles all other general exceptions
        except IndexError:
            text = ":red[Sorry, I can not respond to this prompt...]"
        except Exception as e:
            print(e)
            text = ":red[An error has occured...]"
    
        output_container.write(text)
    return text

# If user provided a new File, create a vector store using that file
# Otherwise, set session_state vector_store to none (For example if user removes the uploaded file)
def set_pdf_state(file: BytesIO | None):
    state_to_store = None
    if file:
        docs = get_docs(file)
        vector_store = create_vector_store(docs)
        state_to_store = vector_store
    st.session_state.pdf_vector_store = state_to_store



# Gemini initilized with top k, top p, and temperature parameters 
# API Key variable is automatically read from .env to ensure security
model = GoogleGenerativeAI(model="gemini-pro", max_retries=6, top_k=10, top_p=0.9, temperature=0.65, verbose=True)

# Two filler columns are placed between col1 and col2 for spacing
# Column one contains the upload button
# Column two contains the clear history button
col1, *_, col2 = st.columns(4, gap="large")
with col1:
    with st.popover("Upload a File"):
        PDFFile = st.file_uploader(label="dds", type=".pdf", label_visibility="collapsed", on_change=on_change_func)
        if st.session_state.pdf_state_changed:
            set_pdf_state(PDFFile)
            st.session_state.pdf_state_changed = False
with col2:
    if st.button("Clear History"):
        st.session_state.history_list = []

# Container that holds the messages 
messages = st.container()
# Chat input UI element
chat = st.chat_input("Enter Text")

# Constantly display the chat history
display_chat_history(st.session_state.history_list)

#If user inputs something in the chat_input, display his message, get and display the AI response, and add both to the history
if chat:
    messages.chat_message("User").write(chat)
    has_context: bool = st.session_state.pdf_vector_store is not None
    current_chain = create_chain(has_context, model)     
        
    current_response = get_response_while_showing_stream(current_chain, chat, st.session_state.history_list, messages)
            

    # Add both the prompt and the response to the history list
    st.session_state.history_list.append(HumanMessage(content=chat))
    st.session_state.history_list.append(AIMessage(content=current_response))
