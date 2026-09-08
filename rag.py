import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Custom RAG Agent", page_icon=":robot_face:", layout="wide")

st.title("Custom RAG Agent")
st.caption("This is a custom RAG agent that can be used to answer questions based on the documents you upload.")


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password",value = os.environ.get("OPENAI_API_KEY", ""))

    model_name = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

    chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=50)

    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)

    top_k = st.slider("Top K", min_value=1, max_value=10, value=3, step=1)


if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf","txt"], accept_multiple_files=True)


def build_vectorstore(pdf_bytes:bytes, size:int, overlap:int) -> FAISS:
    """
    Load the PDF file, split it into chunks, and build a FAISS vector store.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try: docs = PyPDFLoader(tmp_path).load()
    finally: os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=["\n\n", "\n", " ", ""])

    
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

if uploaded_files is not None:
    file_sig =  (uploaded_files.name, uploaded_files.size, chunk_size, chunk_overlap) 

    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("Reading and indexing documents..."):
            st.session_state.vectorstore = build_vectorstore(uploaded_files.read(), chunk_size, chunk_overlap)

        st.session_state.file_sig = file_sig
        st.session_state.messages = []
    st.success("Documents indexed successfully!")

