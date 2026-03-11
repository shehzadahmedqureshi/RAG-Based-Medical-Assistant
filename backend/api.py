import hashlib
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel


load_dotenv()


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "vectorstore", "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


class AskRequest(BaseModel):
    question: str
    # Optional future extension: restrict search to specific files
    file_hashes: Optional[List[str]] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


class UploadResponse(BaseModel):
    status: str  # "processed" | "already_indexed"
    file_name: str
    file_hash: str


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vectorstore() -> Chroma:
    embeddings = get_embedding_model()
    db = Chroma(
        collection_name="medibot_documents",
        embedding_function=embeddings,
        persist_directory=VECTOR_DIR,
    )
    return db


def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def pdf_already_indexed(db: Chroma, file_hash: str) -> bool:
    # Use underlying Chroma collection to query by metadata filter
    collection = db._collection  # type: ignore[attr-defined]
    results = collection.get(where={"file_hash": file_hash}, limit=1)
    ids = results.get("ids") or []
    return len(ids) > 0


def index_pdf(content: bytes, filename: str, file_hash: str) -> None:
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    for i, doc in enumerate(chunks):
        doc.metadata = doc.metadata or {}
        doc.metadata["file_name"] = filename
        doc.metadata["file_hash"] = file_hash
        doc.metadata.setdefault("chunk_index", i)

    db = get_vectorstore()

    # Add in smaller batches to avoid exceeding Chroma's max batch size.
    batch_size = 1000
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        db.add_documents(chunks[start:end])

    db.persist()


def get_rag_chain(db: Chroma):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    groq_model_name = "llama-3.1-8b-instant"
    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api_key,
    )

    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain


app = FastAPI(title="RAG-Based Medical Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_hash = compute_file_hash(content)
    db = get_vectorstore()

    if pdf_already_indexed(db, file_hash):
        return UploadResponse(
            status="already_indexed",
            file_name=file.filename,
            file_hash=file_hash,
        )

    try:
        index_pdf(content, file.filename, file_hash)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc

    return UploadResponse(
        status="processed",
        file_name=file.filename,
        file_hash=file_hash,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    db = get_vectorstore()
    # Quick check for empty vector store
    raw = db._collection.get(limit=1)  # type: ignore[attr-defined]
    if not raw.get("ids"):
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    rag_chain = get_rag_chain(db)
    try:
        result = rag_chain.invoke({"input": payload.question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc

    answer = result.get("answer", "")
    context_docs = result.get("context", []) or []

    source_names: List[str] = []
    seen = set()
    for doc in context_docs:
        meta = getattr(doc, "metadata", {}) or {}
        name = meta.get("file_name") or meta.get("source")
        if name and name not in seen:
            seen.add(name)
            source_names.append(name)

    return AskResponse(answer=answer, sources=source_names)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

