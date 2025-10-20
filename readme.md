# RAG Tutor 

A Retrieval-Augmented Generation (RAG) app that lets you upload PDFs, index them using OpenAI embeddings + FAISS, and query them conversationally through a Streamlit interface.

---

## 🏗 Architecture
- *App.py* — Backend logic (PDF ingestion, FAISS indexing, retrieval, question answering)
- *streamlit.py* — Frontend UI built with Streamlit, imports backend functions from App.py

The backend handles:
- PDF loading
- Chunking & embedding
- FAISS index saving/loading
- Context-aware question answering

The frontend provides:
- PDF upload interface
- API key input
- Log view, chat display
- Interactive Q&A with your documents

---

## 🧩 Running Locally
```bash
pip install -r requirements.txt
streamlit run streamlit.py