import os
import time
import threading
import traceback
import logging
import uuid
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import streamlit as st
from openai import OpenAI

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Directories / Logging Setup
# -----------------------------
APP_DIR = Path(os.getcwd())
UPLOAD_DIR = APP_DIR / "uploaded_pdfs"
INDEXES_DIR = APP_DIR / "faiss_indices"
LOG_DIR = APP_DIR / "logs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "ingest.log"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def _file_log(level: str, msg: str):
    try:
        if level == "debug":
            logging.debug(msg)
        elif level == "info":
            logging.info(msg)
        elif level == "warning":
            logging.warning(msg)
        elif level == "error":
            logging.error(msg)
        elif level == "success":
            logging.info("[SUCCESS] " + msg)
        else:
            logging.info(msg)
    except Exception:
        logging.exception("Failed writing to log file")

# -----------------------------
# Global ingestion state
# -----------------------------
INGEST_STATE = {
    "in_progress": False,
    "messages": deque(maxlen=500),
    "thread": None,
    "result": None,
    "pdf_path": None,
}


def _log(level: str, text: str):
    try:
        INGEST_STATE["messages"].append((level, text))
    except Exception:
        logging.exception("Failed append to INGEST_STATE messages")
    _file_log(level, text)

# -----------------------------
# Backend: ingestion & utilities (UNCHANGED core logic & prompt)
# -----------------------------
def fast_ingest_pipeline(p, api_key, persist_dir_base):
    try:
        # 1) Load PDF
        _log("info", "Loading PDF pages...")
        loader = PyPDFLoader(str(p))
        pages = loader.load()
        _log("info", f"Loaded {len(pages)} pages")

        if len(pages) == 0:
            _log("error", "No pages extracted (possibly image-only PDF).")
            return None

        # 2) Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        _log("info", f"Split into {len(docs)} chunks")

        total_chars = sum(len(d.page_content) for d in docs)
        avg_chars = total_chars / len(docs) if len(docs) else 0
        _log("info", f"Total text length: {total_chars} chars, avg chunk size: {avg_chars:.0f}")

        # 3) Embeddings
        if not api_key:
            _log("error", "OpenAI API key missing.")
            return None

        _log("info", "Preparing OpenAI embeddings...")
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        import tiktoken

        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )

        # ---- Dynamic token-based batching ----
        tokenizer = tiktoken.get_encoding("cl100k_base")

        def estimate_tokens(text: str):
            return len(tokenizer.encode(text))

        MAX_TOKENS_PER_BATCH = 250_000
        batches, current_batch, current_tokens = [], [], 0

        for doc in docs:
            tokens = estimate_tokens(doc.page_content)
            # If adding this doc would exceed limit, start a new batch
            if current_tokens + tokens > MAX_TOKENS_PER_BATCH:
                batches.append(current_batch)
                current_batch, current_tokens = [], 0
            current_batch.append(doc)
            current_tokens += tokens

        if current_batch:
            batches.append(current_batch)

        _log("info", f"Embedding {len(docs)} chunks across {len(batches)} batches")

        faiss_store = None
        for i, batch in enumerate(batches):
            try:
                _log("info", f"Embedding batch {i+1}/{len(batches)} ({len(batch)} chunks)...")
                batch_store = FAISS.from_documents(batch, embeddings)

                if faiss_store is None:
                    faiss_store = batch_store
                else:
                    faiss_store.merge_from(batch_store)

            except Exception as e:
                _log("error", f"Batch {i+1} failed: {e}")
                _log("debug", traceback.format_exc())

        if not faiss_store:
            _log("error", "No embeddings created.")
            return None

        # ---- Save FAISS index ----
        p = Path(p)
        safe_name = p.stem.replace(" ", "_")
        persist_dir = Path(persist_dir_base) / safe_name
        persist_dir.mkdir(parents=True, exist_ok=True)
        faiss_store.save_local(str(persist_dir))
        _log("success", f"FAISS index saved at {persist_dir}")

        retriever = faiss_store.as_retriever(search_kwargs={"k": 4})
        retriever = faiss_store.as_retriever(search_kwargs={"k": 4})

        # ===== ADD DEBUG =====
        _log("debug", f"Retriever type: {type(retriever)}, FAISS store type: {type(faiss_store)}")
        _log("debug", f"Retriever ready: {'Yes' if retriever else 'No'}")
        # =====================

        return retriever, faiss_store

    except Exception as e:
        tb = traceback.format_exc()
        _log("error", f"Ingestion failed: {e}")
        _log("debug", tb)
        return None

def load_faiss_for_pdf(pdf_path: str, api_key: Optional[str] = None, persist_dir_base: Path = INDEXES_DIR) -> Optional[Tuple]:
    p = Path(pdf_path)
    safe_name = p.stem.replace(" ", "_")
    folder = Path(persist_dir_base) / safe_name
    if not folder.exists():
        return None
    try:
        _log("info", f"Loading FAISS index from {folder}")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        faiss_store = FAISS.load_local(str(folder), embeddings)
        retriever = faiss_store.as_retriever(search_kwargs={"k": 4})
        _log("success", "Loaded existing FAISS index.")
        return retriever, faiss_store
    except Exception as e:
        _log("error", f"Failed to load FAISS: {e}")
        _log("debug", traceback.format_exc())
        return None

def ask_question_fast(client_api_key: str, retriever, question: str, model: str = "gpt-3.5-turbo") -> dict:
    """
    Uses the new OpenAI client api (OpenAI.chat.completions.create).
    Builds a context with lightweight citations from retriever docs (if metadata has page/source).
    Prompts the model to answer conversationally, avoid verbatim copying, and include short citations like (Book, p. N).
    """
    try:
        if not client_api_key:
            return {"error": "API key missing."}
        if not retriever:
            return {"error": "Retriever unavailable."}

        # Retrieve relevant docs
        docs = retriever.get_relevant_documents(question)

        # Build context with inline citation markers using metadata if present.
        # Each chunk will be prefixed with a short source note like: [Book, p. 12]
        formatted_chunks = []
        for d in docs[:6]:
            meta = getattr(d, "metadata", {}) or {}
            # Try common metadata fields that may hold page info
            page = meta.get("page") or meta.get("page_number") or meta.get("pageno") or meta.get("source_page")
            source = meta.get("source") or meta.get("source_file") or meta.get("filename")
            citation = None
            if page:
                citation = f"(Book, p. {page})"
            elif source:
                citation = f"({source})"
            else:
                citation = "(Book)"
            # Shorten chunk for safety in context if desired (optional)
            chunk_text = d.page_content.strip()
            formatted_chunks.append(f"{citation}\n{chunk_text}")

        context = "\n\n---\n\n".join(formatted_chunks)

        # Clear, explicit prompt that requests:
        # - conversational tone with a slight "you" touch
        # - avoid verbatim copying
        # - include brief citations (Book, p. N) when info comes from context
        # - if not present in context, say so
        prompt = (
            "You are a knowledgeable, friendly tutor that answers using only the provided textbook excerpts.\n"
            "Answer naturally and conversationally just like a gpt answers (e.g : allright no worries i got you, start like this,dont exactly copy it every time a different start) — add a slight 'you' touch (phrases such as 'you can see', 'it helps to note', etc.).\n"
            "Do NOT copy text verbatim from the context. Paraphrase in your own words.\n"
            "When you use information that appears in the context, add a brief inline citation (for example: (Book page: 12)).\n"
            "Keep citations short and only include them when the context clearly supports the statement after giving the answer in the end in a fancy way.\n"
            "If the answer cannot be found in the context, honestly say: \"I’m not sure — it’s not clearly explained in your uploaded material.\"\n\n"
            f"Context (each chunk labeled with a source):\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer in a clear, slightly conversational tutoring tone, include short citations at the end in a fancy way when appropriate:"
            "if the prompt forces on a very easy response , you have to explain only the mentioned relevent text present in the book to the user in the most simplest words you can"
        )

        # Call new OpenAI client
        client = OpenAI(api_key=client_api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )

        answer = completion.choices[0].message.content.strip()
        return {"result": answer}

    except Exception as e:
        logging.exception("ask_question_fast failed")
        return {"error":str(e)}


