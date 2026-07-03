---
name: rag-pdf-chatbot
description: Build and operate an AI-powered RAG (Retrieval-Augmented Generation) chatbot that ingests PDF documents, indexes them with FAISS vector search, and answers user questions using Groq LLM, served through a Streamlit chat interface. Use this skill when the task involves PDF ingestion, chunking, embedding, vector retrieval, LangChain orchestration, Groq inference, or building/debugging the Streamlit UI for document Q&A.
version: 1.0.0
author: your-github-username
tools: [python, streamlit, langchain, faiss, groq]
---

# RAG PDF Chatbot Skill

## Overview
This skill packages the knowledge needed to build, extend, or debug an AI-powered
Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded
PDF documents. The stack is:

- **LangChain** — document loading, text splitting, retrieval chain orchestration
- **FAISS** — local vector store for similarity search over document embeddings
- **Groq LLM** — fast low-latency inference (e.g., Llama 3.x models) for answer generation
- **Streamlit** — chat-based frontend for uploading PDFs and asking questions

## When to Use This Skill
Apply this skill when the user asks to:
- Set up or modify a PDF ingestion pipeline (loading, chunking, embedding).
- Configure or troubleshoot a FAISS vector index (build, save, load, similarity search).
- Wire up Groq API calls inside a LangChain `RetrievalQA` / `ConversationalRetrievalChain`.
- Build, style, or debug the Streamlit chat UI (file uploader, chat history, streaming responses).
- Deploy the app (e.g., Streamlit Community Cloud, Render, Railway).

## Architecture

```
PDF Upload -> Text Extraction -> Chunking -> Embeddings -> FAISS Index
                                                              |
User Question -> Embed Query -> Similarity Search (top-k) ---+
                                                              |
                                        Retrieved Chunks + Question -> Groq LLM -> Answer
                                                              |
                                                     Streamlit Chat UI (display)
```

## Core Workflow

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install streamlit langchain langchain-community langchain-groq \
            faiss-cpu pypdf sentence-transformers python-dotenv
```
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
```

### 2. PDF Loading and Chunking
- Use `PyPDFLoader` (LangChain) to extract text page-by-page.
- Split with `RecursiveCharacterTextSplitter` — typical settings: `chunk_size=1000`, `chunk_overlap=150`.
- Keep chunk size small enough to fit context window but large enough to preserve meaning.

### 3. Embeddings and FAISS Index
- Use a local embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2` via
  `HuggingFaceEmbeddings`) to avoid extra API cost, since Groq does not serve embeddings.
- Build the index with `FAISS.from_documents(chunks, embeddings)`.
- Persist locally with `vectorstore.save_local("faiss_index")` and reload with
  `FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)`.

### 4. Retrieval + Groq Generation Chain
- Instantiate the LLM: `ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=..., temperature=0)`.
- Build a retriever: `vectorstore.as_retriever(search_kwargs={"k": 4})`.
- Wrap in `ConversationalRetrievalChain.from_llm(llm, retriever, memory=...)` to support
  multi-turn chat with `ConversationBufferMemory`.
- Add a system prompt instructing the model to answer only from retrieved context and
  say "I don't know" when the answer isn't in the document.

### 5. Streamlit Chat Interface
- Use `st.file_uploader` for PDF upload, cache the processed index with
  `st.session_state` so re-uploads aren't reprocessed unnecessarily.
- Use `st.chat_message` and `st.chat_input` for the conversational UI.
- Stream Groq responses token-by-token if using `.stream()` for a snappier feel.

### 6. Deployment
- For quick hosting: Streamlit Community Cloud (free, direct GitHub deploy).
- For more control/backend separation: Render or Railway, matching the user's existing
  deployment workflow with Vercel/Render/Railway.
- Store `GROQ_API_KEY` as an environment secret on the hosting platform, never commit it.

## Common Pitfalls
- FAISS index built with one embedding model is incompatible with another — always
  rebuild if you switch embedding models.
- Large PDFs can exceed Groq's context window if too many chunks (k) are retrieved;
  keep k between 3-6 for most use cases.
- Streamlit reruns the whole script on every interaction — guard expensive steps
  (PDF processing, index building) with `st.session_state` checks or `@st.cache_resource`.
- Groq rate limits are stricter on free tier; add retry/backoff logic for production use.

## File Structure Reference
```
rag-pdf-chatbot/
├── app.py                # Streamlit entrypoint
├── rag_pipeline.py       # Loading, chunking, embedding, FAISS logic
├── requirements.txt
├── .env                  # GROQ_API_KEY (not committed)
├── faiss_index/          # Persisted vector store (gitignored)
└── README.md
```

## Related Skills
- General LangChain orchestration
- Streamlit UI/UX patterns
- Vector database tuning (FAISS, Chroma, Pinecone)
