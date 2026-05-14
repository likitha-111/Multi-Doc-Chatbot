from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import shutil
import tempfile
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Document Knowledge Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    session_id: str = Field(default="default", description="Session ID for memory")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key (falls back to GROQ_API_KEY env var)")

class Source(BaseModel):
    filename: str
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str

class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks: int

class DocumentInfo(BaseModel):
    filename: str
    chunks: int

class DocumentsListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int

class ClearResponse(BaseModel):
    message: str

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore: Optional[Chroma] = None
session_memories: dict = {}

def get_or_create_vectorstore() -> Chroma:
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            collection_name="documents",
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )
    return vectorstore

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    return session_memories[session_id]


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an expert document analyst and knowledge assistant.
Your task is to answer questions based ONLY on the provided document context.

Chat History:
{chat_history}

Document Context:
{context}

Question: {question}

Instructions:
- Answer based strictly on the provided context
- Be thorough, accurate, and well-structured
- If the answer isn't in the context, clearly state "I couldn't find this information in the uploaded documents"
- Always cite using only numbered bracket references like [1], [2], etc.
- Do not add extra attribution phrases such as "According to the provided document context" or "specifically in..."
- NEVER include cross-references, forward references, or meta-commentary about other lectures, chapters, sections, or documents. Do not write lines like "This is also covered in Lecture-X", "See Chapter Y for details"
- Use bullet points or numbered lists when appropriate for clarity

Answer:""",
)


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents into the vector store."""
    vs = get_or_create_vectorstore()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    processed_files = []
    total_chunks = 0

    for upload_file in files:
        if not upload_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload_file.filename} is not a PDF")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Tag each page with its source filename
            for page in pages:
                page.metadata["source_filename"] = upload_file.filename

            chunks = splitter.split_documents(pages)
            vs.add_documents(chunks)
            total_chunks += len(chunks)
            processed_files.append(upload_file.filename)
            logger.info(f"Processed {upload_file.filename}: {len(chunks)} chunks")
        finally:
            os.unlink(tmp_path)

    return UploadResponse(
        message=f"Successfully processed {len(processed_files)} document(s)",
        files_processed=processed_files,
        total_chunks=total_chunks,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the documents using conversational retrieval."""
    vs = get_or_create_vectorstore()

    try:
        count = vs._collection.count()
    except Exception:
        count = 0

    if count == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload PDFs first.")

    resolved_key = request.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not resolved_key:
        raise HTTPException(status_code=400, detail="Groq API key not provided and GROQ_API_KEY env var is not set.")

    llm = ChatGroq(
        groq_api_key=resolved_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=2048,
    )

    memory = get_session_memory(request.session_id)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=False,
    )

    result = chain.invoke({"question": request.question})
    answer = result["answer"]
    source_docs = result.get("source_documents", [])

    # De-duplicate sources
    seen = set()
    sources: List[Source] = []
    for doc in source_docs:
        filename = doc.metadata.get("source_filename", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0) + 1
        snippet = doc.page_content[:200].strip().replace("\n", " ")
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(filename=filename, page=page, snippet=snippet))

    return ChatResponse(answer=answer, sources=sources, session_id=request.session_id)


@app.get("/documents", response_model=DocumentsListResponse)
async def list_documents():
    """List all uploaded documents and their chunk counts."""
    vs = get_or_create_vectorstore()
    try:
        results = vs._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
    except Exception:
        metadatas = []

    doc_chunks: dict = {}
    for meta in metadatas:
        fname = meta.get("source_filename", meta.get("source", "Unknown"))
        doc_chunks[fname] = doc_chunks.get(fname, 0) + 1

    documents = [DocumentInfo(filename=k, chunks=v) for k, v in doc_chunks.items()]
    return DocumentsListResponse(documents=documents, total_chunks=sum(doc_chunks.values()))


@app.delete("/clear", response_model=ClearResponse)
async def clear_documents():
    """Clear all documents from the vector store and reset memories."""
    global vectorstore, session_memories
    if vectorstore:
        vectorstore.delete_collection()
        vectorstore = None
    session_memories = {}
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return ClearResponse(message="All documents and chat history cleared successfully")


@app.delete("/session/{session_id}", response_model=ClearResponse)
async def clear_session(session_id: str):
    """Clear chat history for a specific session."""
    if session_id in session_memories:
        del session_memories[session_id]
    return ClearResponse(message=f"Session {session_id} cleared")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Multi-Document Knowledge Assistant"}