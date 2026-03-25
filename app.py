import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("PDF based Chatbot")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# API key
api_key = st.text_input("Enter Groq API Key", type="password")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based only on the context.

Context:
{context}

Chat History:
{history}

Question:
{input}
""")

# RAG function
def generate_answer(query, retriever, llm):
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )

    final_prompt = prompt.format(
        context=context,
        history=history,
        input=query
    )

    response = llm.invoke(final_prompt)

    return response.content


# Process PDF only once
if uploaded_file and api_key:

    if "retriever" not in st.session_state:

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(docs, embeddings)
        st.session_state.retriever = db.as_retriever()

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

    st.success("✅ PDF processed! Start chatting below")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        answer = generate_answer(
            user_input,
            st.session_state.retriever,
            llm
        )

        # Store bot response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.write(answer)