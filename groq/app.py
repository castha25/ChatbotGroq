import streamlit as st
import os
import time
import tempfile

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader
)

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------
# API KEY
# --------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

# --------------------------------
# EMBEDDINGS
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------
# TITLE
# --------------------------------

st.title("📚 RAG Chatbot (Groq)")

# --------------------------------
# SOURCE SELECTION
# --------------------------------

source = st.radio(
    "Choose Knowledge Source",
    ["LangSmith Docs", "Upload PDF"]
)

# --------------------------------
# BUILD VECTOR STORE
# --------------------------------

if source == "Url":

    if "docs_vectors" not in st.session_state:

        urls = [
            "https://docs.smith.langchain.com/",
            "https://docs.langchain.com/langsmith/observability/",
            "https://docs.langchain.com/langsmith/engine-overview/"
        ]

        loader = WebBaseLoader(urls)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400
        )

        final_documents = splitter.split_documents(docs)

        st.write(f"Documents Loaded: {len(docs)}")
        st.write(f"Chunks Created: {len(final_documents)}")

        vectors = FAISS.from_documents(
            final_documents,
            embeddings
        )

        st.session_state.docs_vectors = vectors

    vectorstore = st.session_state.docs_vectors

# --------------------------------
# PDF RAG
# --------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf"
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400
        )

        final_documents = splitter.split_documents(docs)

        st.write(f"Pages Loaded: {len(docs)}")
        st.write(f"Chunks Created: {len(final_documents)}")

        vectorstore = FAISS.from_documents(
            final_documents,
            embeddings
        )

    else:
        vectorstore = None

# --------------------------------
# LLM
# --------------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# --------------------------------
# PROMPT
# --------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY from the provided context.

Use all relevant information.

If the answer is available:
- Explain in detail.
- Include steps when applicable.
- Do not shorten unnecessarily.

If the answer is not present in the context,
clearly say so.

Context:
{context}

Question:
{question}
"""
)

# --------------------------------
# QUESTION INPUT
# --------------------------------

user_question = st.text_input(
    "Ask your question"
)

# --------------------------------
# RAG PIPELINE
# --------------------------------

if user_question and vectorstore:

    start = time.process_time()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 15}
    )

    docs = retriever.invoke(user_question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": user_question
        }
    )

    response = llm.invoke(formatted_prompt)

    st.subheader("Answer")

    st.write(response.content)

    st.write(
        f"Response Time: {time.process_time() - start:.2f} seconds"
    )

    with st.expander("Retrieved Documents"):

        for i, doc in enumerate(docs, start=1):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)

            st.markdown("---")