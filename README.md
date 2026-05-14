# 🧠 RAG Assistant — Multi-Document Knowledge Chat

A Streamlit + FastAPI app for uploading PDFs, building a local vector knowledge base, and asking conversational questions powered by Groq.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
│                   Streamlit Frontend :8501                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────┐
│                FastAPI Backend :8001                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  /upload     │  │  /chat       │  │  /documents      │   │
│  │  PyPDFLoader │  │  Groq LLM    │  │  /clear          │   │
│  │  TextSplitter│  │  ConvChain   │  │  /session/:id    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘   │
└─────────┼─────────────────┼─────────────────────────────────┘
          │                 │
┌─────────▼─────────────────▼──────────────────────────────────┐
│              LangChain + Chroma + Groq                       │
│  ┌────────────────────┐    ┌─────────────────────────────┐   │
│  │ HuggingFace        │    │ ConversationalRetrievalChain│   │
│  │ Sentence           │    │ ConversationBufferMemory    │   │
│  │ Transformers       │    │ MMR Retrieval (k=5)         │   │
│  │ all-MiniLM-L6-v2   │    │ Custom QA Prompt            │   │
│  └────────┬───────────┘    └─────────────────────────────┘   │
└───────────┼──────────────────────────────────────────────────┘
            │ embeddings
┌───────────▼──────────────────────────────────────────────────┐
│                   ChromaDB (local persist)                   │
│              ./data/chroma_db/                               │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Multi-PDF Upload** | Upload any number of PDFs at once via drag & drop |
| **Semantic Embeddings** | `all-MiniLM-L6-v2` via Sentence Transformers — runs locally, no API cost |
| **Vector Store** | ChromaDB with persistent storage across restarts |
| **Conversational Memory** | Per-session `ConversationBufferMemory` — remembers prior turns |
| **Source Citation** | Every answer shows which file + page the info came from |
| **MMR Retrieval** | Maximal Marginal Relevance for diverse, non-redundant context |
| **Fast Inference** | Groq API with `llama-3.3-70b-versatile` for near-instant answers |
| **Session Management** | Multiple isolated chat sessions; clear anytime |
| **Pydantic v2 Models** | Strict request/response validation throughout |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file or export the variables directly:

```bash
GROQ_API_KEY=your_groq_api_key_here
API_BASE_URL=http://localhost:8001
```

The frontend loads `GROQ_API_KEY` from the environment and sends requests to `API_BASE_URL`.

### 3. Start the backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Start the frontend

```bash
streamlit run app.py --server.port 8501
```

### 5. Open the app

→ **http://localhost:8501**

---

## 📖 Usage

1. Confirm `GROQ_API_KEY` is set in your environment.
2. Upload one or more PDF files with the file uploader.
3. Click **Process Documents** to embed and index the PDFs.
4. Ask questions in the chat input.
5. Use **New Chat** to reset conversational memory for the current session.
6. Use **Clear DB** to wipe the indexed documents and restart fresh.

---

## 🔧 API Reference

The FastAPI backend exposes:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload one or more PDF files |
| `POST` | `/chat` | Send a question, get an answer + sources |
| `GET` | `/documents` | List indexed documents and chunk counts |
| `DELETE` | `/clear` | Wipe all documents and sessions |
| `DELETE` | `/session/{id}` | Clear a specific chat session |
| `GET` | `/health` | Health check |

Interactive API docs: **http://localhost:8001/docs**

---

## 📦 Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Frontend | Streamlit | Web UI |
| Backend | FastAPI | REST API |
| Validation | Pydantic | Request/response models |
| Embeddings | Sentence Transformers | `all-MiniLM-L6-v2` embeddings |
| Vector Store | ChromaDB | Local semantic search |
| PDF Parsing | PyPDF | Document loading |
| LLM | Groq | `llama-3.3-70b-versatile` chat generation |

---

## ⚙️ Configuration

`main.py` contains the backend configuration:

- Embedding model: `all-MiniLM-L6-v2`
- LLM model: `llama-3.3-70b-versatile`
- Chunk size: `1000`
- Chunk overlap: `200`
- Retriever: MMR with `k=5`, `fetch_k=10`

The frontend uses `app.py` and expects the backend at `API_BASE_URL`.

---

## 📁 Project Structure

```
.
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .env
```
