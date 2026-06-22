import streamlit as st
import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -----------------------------
# API Key
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# Build Vector Store Once
# -----------------------------
if "vectors" not in st.session_state:

    embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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
    print("Documents loaded:", len(docs))
    print("Chunks created:", len(final_documents))

    vectors = FAISS.from_documents(
        final_documents,
        embeddings
    )

    st.session_state.embeddings = embeddings
    st.session_state.vectors = vectors

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question only from the provided context.


Use all relevant information from the context.

If the answer is found:
- Explain in detail.
- Include important steps and concepts.
- Do not unnecessarily shorten the answer.

If the context does not contain enough information, clearly state that.

Context:
{context}

Question:
{question}
"""
)


# -----------------------------
# UI
# -----------------------------
st.title("RAG ChatbotGroq ")

user_question = st.text_input(
    "Input your prompt here"
)

if user_question:

    start = time.process_time()

    retriever = st.session_state.vectors.as_retriever(
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

            st.markdown(f"### Document Chunk {i}")

            st.write(doc.page_content)

            st.markdown("---")