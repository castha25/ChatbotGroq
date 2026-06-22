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

# --------------------------------
# LOAD ENV
# --------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="ChatbotGroq",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ChatbotGroq")
st.markdown(
    "Chat with content from **multiple URLs** or an **uploaded PDF** using RAG."
)

# --------------------------------
# EMBEDDINGS
# --------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

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

Use all relevant information from the context.

If the answer is available:
- Explain in detail.
- Include important steps if applicable.
- Do not shorten unnecessarily.

If the answer is not available in the context,
clearly state that.

Context:
{context}

Question:
{question}
"""
)

# --------------------------------
# SOURCE SELECTION
# --------------------------------

source = st.radio(
    "Choose Knowledge Source",
    ["URL", "Upload PDF"]
)

vectorstore = None

# --------------------------------
# URL MODE
# --------------------------------

if source == "URL":

    urls_text = st.text_area(
        "Enter URLs (one URL per line)",
        height=150,
        placeholder="""https://docs.smith.langchain.com/
https://docs.langchain.com/langsmith/observability/
https://docs.langchain.com/langsmith/engine-overview/"""
    )

    if urls_text:

        urls = [
            url.strip()
            for url in urls_text.split("\n")
            if url.strip()
        ]

        if (
            "docs_vectors" not in st.session_state
            or st.session_state.get("loaded_urls") != urls
        ):

            with st.spinner("Loading URLs and creating vector database..."):

                loader = WebBaseLoader(urls)

                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=400
                )

                final_documents = splitter.split_documents(docs)

                st.success(
                    f"Loaded {len(docs)} documents and created {len(final_documents)} chunks."
                )

                vectors = FAISS.from_documents(
                    final_documents,
                    embeddings
                )

                st.session_state.docs_vectors = vectors
                st.session_state.loaded_urls = urls

        vectorstore = st.session_state.docs_vectors

# --------------------------------
# PDF MODE
# --------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        with st.spinner("Processing PDF..."):

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

            st.success(
                f"Loaded {len(docs)} pages and created {len(final_documents)} chunks."
            )

            vectorstore = FAISS.from_documents(
                final_documents,
                embeddings
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

if user_question and vectorstore is not None:

    start = time.process_time()

    with st.spinner("Generating answer..."):

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 15}
        )

        retrieved_docs = retriever.invoke(
            user_question
        )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in retrieved_docs
            ]
        )

        formatted_prompt = prompt.invoke(
            {
                "context": context,
                "question": user_question
            }
        )

        response = llm.invoke(
            formatted_prompt
        )

    st.subheader("Answer")

    st.write(response.content)

    st.info(
        f"Response Time: {time.process_time() - start:.2f} seconds"
    )

    with st.expander("Retrieved Documents"):

        for i, doc in enumerate(
            retrieved_docs,
            start=1
        ):

            st.markdown(
                f"### Chunk {i}"
            )

            st.write(
                doc.page_content
            )

            st.markdown("---")