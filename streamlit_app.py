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




load_dotenv()
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
if "pdf_vector_store" not in st.session_state:
    st.session_state.pdf_vector_store = None

# Function that goes through the history list and display the chat history
def display_chat_history(history:list[BaseMessage]):
    user_message = True
    for message in history:
        message_sender = "User" if user_message else "AI"
        messages.chat_message(message_sender).write(message.content)
        user_message = not user_message

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
    st.session_state.pdf_vector_store = get_pdf_content(PDF)
    if len(st.session_state.pdf_vector_store) > MAX_PROMPT_CHAR_COUNT:
        st.warning("PDF length is greater than supported length. Prompt might not work...")

# Will get called when the user changes the state of the st.file_uploader object
def on_change_func():
    st.session_state.pdf_state_changed = True


def get_docs(file: BytesIO):
    loader = BytesIOPyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def create_vector_store(docs):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embedding=embedding)
    return vector_store

def create_chain(with_context: bool, model):
    messages_list = [
        ("system", "Answer the user's questions in a friendly and respectful manner, and make sure to NOT prefix your responses with AI: or System:"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ]
    if with_context:
        messages_list[0] = ("system", "Answer the user's questions based on the context: {context}")
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
    else:
        prompt = ChatPromptTemplate.from_messages(messages_list)
        chain = prompt | model
        return chain


    
    
 


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
        # two exceptions may occur
        except IndexError:
            # this error is when the response is restricted (based on Google's terms and conditions)
            text = ":red[Sorry, I can not respond to this prompt...]"
        except Exception as e:
            print(e)
            # general exceptions
            text = ":red[An error has occured...]"
    
        output_container.write(text)
    return text

def set_pdf_state(file: BytesIO | None):
    state_to_store = None
    if file:
        docs = get_docs(file)
        vector_store = create_vector_store(docs)
        state_to_store = vector_store
    st.session_state.pdf_vector_store = state_to_store


# A session key and title is hard coded for program simplicity and to avoid user confusion
st.title = "ChatGemini"

# Gemini initilized with top k, top p, and temperature parameters 
model = GoogleGenerativeAI(model="gemini-pro", max_retries=6, top_k=10, top_p=0.9, temperature=0.65, verbose=True)

# Two filler columns are placed between col1 and col2 for spacing
col1, *_, col2 = st.columns(4, gap="large")
with col1:
    with st.popover("Upload a File"):
        PDFFile = st.file_uploader(label="dds", type=".pdf", label_visibility="collapsed", on_change=on_change_func)
        if st.session_state.pdf_state_changed:
            set_pdf_state(PDFFile)
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
    has_context: bool = st.session_state.pdf_vector_store is not None
    current_chain = create_chain(has_context, model)     
        
    current_response = get_response_while_showing_stream(current_chain, chat, st.session_state.stack, messages)
            

    # adding Gemini's response to the history stack
    st.session_state.stack.append(HumanMessage(content=chat))
    st.session_state.stack.append(AIMessage(content=current_response))
