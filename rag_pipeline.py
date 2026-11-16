"""
rag_pipeline.py — Multi-Document RAG Chatbot (Core Logic)
==========================================================
Embeddings : HuggingFace all-MiniLM-L6-v2  (local, free, no quota)
LLM        : Groq  — llama-3.1-8b-instant  (fast, generous free tier)
Vector DB  : ChromaDB (persisted on disk in ./chroma_db)
"""

import os
import time
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 150
TOP_K_DOCS         = 5

# Try models in this order — all on Groq free tier (30 req/min, 14 400/day)
LLM_MODEL_FALLBACKS = [
    "llama-3.1-8b-instant",       # fastest, lowest token count
    "llama-3.3-70b-versatile",    # larger, if small one is rate-limited
]

MAX_RETRIES_PER_MODEL = 2

RAG_PROMPT_TEMPLATE = """You are a precise document assistant. Answer the user's question
using ONLY the provided document excerpts below. If the answer is not in the excerpts,
say "I couldn't find this in the uploaded documents."

Be concise, factual, and always mention which document/page each piece of information came from.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


# ──────────────────────────────────────────────
# PDF Ingestion
# ──────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF and tag every page with source filename + 1-indexed page number."""
    loader = PyPDFLoader(file_path)
    pages  = loader.load()
    doc_name = Path(file_path).name
    for page in pages:
        page.metadata["source"] = doc_name
        page.metadata["page"]   = page.metadata.get("page", 0) + 1
    return pages


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split pages into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def file_hash(file_path: str) -> str:
    """MD5 hash of a file — used to avoid re-indexing the same PDF twice."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ──────────────────────────────────────────────
# Embeddings & Vector Store
# ──────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Local HuggingFace embeddings — no API key, no quota, works completely offline."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_or_create_vectorstore(api_key: str = "") -> Chroma:
    """Open (or create) a persisted ChromaDB collection."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="rag_docs",
    )


def add_documents_to_store(
    vectorstore: Chroma, chunks: List[Document], doc_name: str
) -> int:
    """Add chunks to ChromaDB and return the number of chunks indexed."""
    for chunk in chunks:
        chunk.metadata["doc_name"] = doc_name
    vectorstore.add_documents(chunks)
    return len(chunks)


def get_indexed_docs(vectorstore: Chroma) -> List[str]:
    """Return a sorted list of document names currently in the vector store."""
    try:
        results = vectorstore._collection.get(include=["metadatas"])
        names   = {m.get("doc_name", "unknown") for m in results["metadatas"] if m}
        return sorted(names)
    except Exception:
        return []


def delete_document(vectorstore: Chroma, doc_name: str) -> int:
    """Delete all chunks belonging to a given document name."""
    try:
        results       = vectorstore._collection.get(include=["metadatas"])
        ids_to_delete = [
            results["ids"][i]
            for i, m in enumerate(results["metadatas"])
            if m.get("doc_name") == doc_name
        ]
        if ids_to_delete:
            vectorstore._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────

def get_llm(api_key: str, model: str) -> ChatGroq:
    return ChatGroq(model=model, groq_api_key=api_key, temperature=0.2)


# ──────────────────────────────────────────────
# RAG Query (Retrieve → Prompt → Generate)
# ──────────────────────────────────────────────

def query_rag(
    vectorstore: Chroma,
    api_key: str,
    question: str,
) -> Tuple[str, List[Dict]]:
    """
    Full RAG pipeline:
      1. MMR retrieval (diverse top-K chunks)
      2. Build context string with source labels
      3. Call Groq LLM — with automatic model fallback + rate-limit retry
      4. Return (answer, citations)
    """

    # Step 1 — Retrieve
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_DOCS, "fetch_k": TOP_K_DOCS * 2},
    )
    docs = retriever.invoke(question)

    if not docs:
        return "I couldn't find this in the uploaded documents.", []

    # Step 2 — Build context
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        context_parts.append(f"[{source} · Page {page}]:\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    prompt  = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # Step 3 — Generate with fallback + retry
    answer     = None
    last_error = None

    for model_name in LLM_MODEL_FALLBACKS:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                llm      = get_llm(api_key, model_name)
                response = llm.invoke([HumanMessage(content=prompt)])
                answer   = response.content.strip()
                break                           # success → exit retry loop
            except Exception as e:
                last_error = e
                err = str(e)

                # Rate-limited (429 / RESOURCE_EXHAUSTED) — wait then retry same model
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                    match = (
                        re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", err)
                        or re.search(r"try again in (\d+(?:\.\d+)?)s", err, re.IGNORECASE)
                    )
                    wait = min(float(match.group(1)) if match else 15, 30)
                    time.sleep(wait)
                    continue                    # retry same model

                # Model unavailable (404) — skip to next model immediately
                if "404" in err or "NOT_FOUND" in err:
                    break

                break                           # unknown error — skip to next model

        if answer is not None:
            break                               # got answer → stop trying other models

    if answer is None:
        real_error = str(last_error) if last_error else "Unknown error"
        return (
            f"⚠️ Groq API error: {real_error[:300]}\n\n"
            "Check your GROQ_API_KEY in Streamlit Secrets.",
            [],
        )

    # Step 4 — Build citations (deduplicated by source + page)
    citations = []
    seen = set()
    for doc in docs:
        meta = doc.metadata
        key  = (meta.get("source", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            citations.append({
                "source":  meta.get("source", "Unknown"),
                "page":    meta.get("page", "?"),
                "snippet": doc.page_content[:250].replace("\n", " ").strip() + "…",
            })

    return answer, citations