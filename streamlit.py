import os
import time
import threading
import traceback
import uuid
from pathlib import Path
import logging

import streamlit as st

from App import (
    LOG_PATH,
    UPLOAD_DIR,
    INDEXES_DIR,
    INGEST_STATE,
    _log,
    fast_ingest_pipeline,
    load_faiss_for_pdf,
    ask_question_fast,
)







# Streamlit App (merged UI)
# -----------------------------
def main():
    st.set_page_config(page_title="RAG Tutor — Stable", page_icon="📘", layout="wide")

    # Session defaults
    for key, val in {
        "show_about": False,
        "show_help": False,
        "show_contact": False,
        "chat_history": [],
        "book_loaded": False,
        "user_api_key": None,
        "_rerun_done": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # -----------------------------
    # Styling (Royal/Lavender, top bar)
    # -----------------------------
    st.markdown(
        """
<style>
:root{
  --navy: #0B1F3B;
  --lavender: #E6D6FF;
  --button-royal: #123A6B;
  --button-text: #F3E9FF;
  --text-dark: #0B1726;
}
body { background-color: var(--navy); }

div.block-container {
  background-color: var(--lavender);
  border-radius: 14px;
  padding: 2rem 2.5rem;
  margin-top: 18px;
}

[data-testid='stSidebar'] {
  background-color: var(--navy);
  color: var(--lavender);
  padding: 2rem 1.25rem;
}
[data-testid='stSidebar'] label,
[data-testid='stSidebar'] .stTextInput label {
  color: #ffffff !important;
  font-weight:700;
}
[data-testid='stSidebar'] input { color:#000000; }

/* Topbar */
.topbar {
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  margin-bottom: 14px;
}
.rag-title {
  color: var(--button-text);
  font-size: 44px;
  font-weight: 800;
  text-align: center;
  letter-spacing: 0.4px;
}

/* Top-right buttons */
.top-controls {
  position: absolute;
  right: 25px;
  top: 5px;
  display: flex;
  gap: 8px;
}
.ctrl-btn {
  background-color: var(--button-royal);
  color: var(--button-text);
  border: none;
  padding: 8px 14px;
  border-radius: 10px;
  font-weight: 600;
  cursor: pointer;
}
.ctrl-btn:hover { opacity: 0.95; }

.info-box {
  background: rgba(255,255,255,0.07);
  padding: 14px 16px;
  border-radius: 10px;
  color: var(--text-dark);
  margin-top: 8px;
}

/* Small responsive tweaks */
@media (max-width: 800px) {
  .rag-title { font-size: 28px; }
  .top-controls { right: 10px; top: -4px; }
}
</style>
""",
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Header Section
    # -----------------------------
    st.markdown("<div class='topbar'>", unsafe_allow_html=True)
    st.markdown("<h1 class='rag-title'>📚 RAG Tutor 🧠</h1>", unsafe_allow_html=True)

    # Top-right controls implemented via Streamlit buttons (toggles)
    cols = st.columns([1, 0.2])
    with cols[1]:
        # Use compact buttons in a horizontal layout
        about_clicked = st.button("About", key="btn_about")
        contact_clicked = st.button("Contact", key="btn_contact")
        help_clicked = st.button("Help", key="btn_help")

        if about_clicked:
            st.session_state.show_about = not st.session_state.show_about
            # ensure other modals closed
            st.session_state.show_help = False
            st.session_state.show_contact = False
        if contact_clicked:
            st.session_state.show_contact = not st.session_state.show_contact
            st.session_state.show_about = False
            st.session_state.show_help = False
        if help_clicked:
            st.session_state.show_help = not st.session_state.show_help
            st.session_state.show_about = False
            st.session_state.show_contact = False

    st.markdown("</div>", unsafe_allow_html=True)

    # Show About / Help / Contact as expanders / info boxes
    if st.session_state.show_about:
        st.markdown(
            """
            <div class='info-box'>
            <strong>About RAG Tutor</strong><br>
            RAG Tutor build by :
            Umer Iqbal
            
        
            """,
            unsafe_allow_html=True,
        )
    if st.session_state.show_help:
        st.markdown(
            """
            <div class='info-box'>
            <strong>Help</strong><br>
            - Use a text-based (not scanned) PDF for best results.<br>
            - Paste your OpenAI API key in the sidebar before indexing.<br>
            - If the model cannot find the answer in your book, it will say so honestly.
            </div>
            """,
            unsafe_allow_html=True,
        )
    if st.session_state.show_contact:
        st.markdown(
            """
            <div class='info-box'>
            <strong>Contact</strong><br>
            For issues, contact the developer:  umeriql18@gmail.com
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Quick instructions
    st.info(
        "Quick Instructions  \n"
        "- Enter your OpenAI API key in the sidebar before uploading.\n"
        "- Upload a text-based (not scanned) PDF.\n"
        "- Click '🚀 Build Knowledge Index' to build index, then ask questions.",
        icon="ℹ",
    )

    # Sidebar (Settings + Logs) - keep behavior from stable backend but style from UI
    with st.sidebar:
        st.header("⚙ Settings & Logs")
        api_key_input = st.text_input("OpenAI API Key", type="password", key="api_key_input")
        if api_key_input:
            st.session_state.user_api_key = api_key_input
            st.success("✅ API Key saved")
        elif st.session_state.get("user_api_key"):
            st.info("✅ Using saved API Key")

        st.markdown("---")
        st.subheader("Logs (last 60 lines)")
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()[-60:]
                st.text("".join(lines))
        except Exception:
            st.info("No logs yet.")

    # Main layout: left content + right debug/chat
    col1, col2 = st.columns([2, 1])

    # -----------------------------
    # Left column: Upload, Ingest, Ask
    # -----------------------------
    with col1:
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")

        if uploaded_file is not None and "uploaded_filepath" not in st.session_state:
            # Unique subfolder per upload
            unique_dir = UPLOAD_DIR / f"{Path(uploaded_file.name).stem}_{uuid.uuid4().hex[:8]}"
            unique_dir.mkdir(parents=True, exist_ok=True)
            save_path = unique_dir / uploaded_file.name

            with st.spinner("💾 Saving PDF..."):
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        f.flush()
                        os.fsync(f.fileno())

                    st.session_state.uploaded_filepath = str(save_path)
                    st.session_state.uploaded_name = uploaded_file.name
                    _log("info", f"Saved PDF: {save_path}")
                    st.success(f"✅ Saved at {save_path}")
                except Exception as e:
                    st.error(f"Failed to save PDF: {e}")
                    logging.exception("Failed saving PDF")

        if st.session_state.get("uploaded_filepath"):
            p = Path(st.session_state.uploaded_filepath)
            if p.exists():
                st.info(f"📁 File ready: {p}")
            else:
                st.error("❌ File missing! Re-upload required.")

        st.markdown("---")
        st.subheader("🚀 Build Knowledge Index")

        with st.expander("Ingestion Log"):
            msgs = list(INGEST_STATE["messages"])
            if not msgs:
                st.info("No activity yet.")
            else:
                for lvl, txt in msgs[-200:]:
                    color = (
                        "red" if lvl == "error"
                        else "orange" if lvl == "warning"
                        else "green" if lvl == "success"
                        else "white"
                    )
                    st.markdown(f"<span style='color:{color}'>• {txt}</span>", unsafe_allow_html=True)

        # Show a running notice if ingestion in progress
        if INGEST_STATE.get("in_progress", False):
            st.warning("⚙ Ingestion running... Please wait.")
        else:
            # Build Index Button
            if st.button("🚀 Build Knowledge Index", use_container_width=True, type="primary"):
                if not st.session_state.get("uploaded_filepath"):
                    st.error("❌ Upload a PDF first.")
                else:
                    pdf_path = st.session_state.uploaded_filepath
                    api_key = st.session_state.get("user_api_key") or os.getenv("OPENAI_API_KEY")

                    # Clear rerun fuse for fresh ingestion
                    st.session_state._rerun_done = False

                    # Try loading existing FAISS index
                    existing = load_faiss_for_pdf(pdf_path, api_key=api_key)
                    if existing:
                        st.session_state.retriever, st.session_state.faiss_store = existing
                        st.session_state.book_loaded = True
                        _log("success", "✅ Loaded existing FAISS index")
                        # One-time rerun refresh (guarded)
                        if not st.session_state.get("_rerun_done"):
                            st.session_state._rerun_done = True
                            if hasattr(st, "rerun"):
                                st.rerun()
                        st.stop()

                    # -----------------------------
                    # Background ingestion thread
                    # -----------------------------
                    def _thread_target():
                        try:
                            INGEST_STATE["in_progress"] = True
                            # reset messages so UI shows the fresh log stream
                            INGEST_STATE["messages"].clear()
                            INGEST_STATE["result"] = None
                            _log("info", "Starting ingestion pipeline...")
                            res = fast_ingest_pipeline(pdf_path, api_key=api_key, persist_dir_base=INDEXES_DIR)
                            if res:
                                retriever, faiss_store = res
                                # save the result to the shared state (thread-safe)
                                INGEST_STATE["result"] = (retriever, faiss_store)
                                _log("success", "✅ Ingestion pipeline finished successfully.")
                            else:
                                _log("error", "❌ Ingestion failed — no retriever returned.")
                        except Exception as e:
                            _log("error", f"Thread exception: {e}")
                            _log("debug", traceback.format_exc())
                        finally:
                            # Set in_progress False last so main thread sees result if present
                            INGEST_STATE["in_progress"] = False
                            _log("info", "Ingestion thread done.")

                    # Start ingestion thread
                    t = threading.Thread(target=_thread_target, daemon=True)
                    INGEST_STATE["thread"] = t
                    t.start()

                    # --- Poll while ingestion runs and show spinner the whole time ---
                    with st.spinner("⚙ Building knowledge index... this may take a while"):
                        # Poll every 1 second (non-busy)
                        while INGEST_STATE.get("in_progress", False):
                            time.sleep(1)
                        # When we exit the loop, ingestion thread has finished (in_progress == False)

                    # At this point the background thread has finished (or failed).
                    if INGEST_STATE.get("result"):
                        try:
                            retriever, faiss_store = INGEST_STATE["result"]
                            st.session_state.retriever = retriever
                            st.session_state.faiss_store = faiss_store
                            st.session_state.book_loaded = True
                            _log("success", "✅ Retriever loaded into session_state successfully.")
                            st.success("✅ Knowledge index built and ready for questions!")

                            # One-time rerun to refresh UI (guarded)
                            if not st.session_state.get("_rerun_done"):
                                st.session_state._rerun_done = True
                                if hasattr(st, "rerun"):
                                    st.rerun()
                            st.stop()
                        except Exception as e:
                            _log("error", f"Error while loading retriever: {e}")
                            st.error(f"Error loading retriever: {e}")
                    else:
                        # Ingestion finished but returned no result; user should check logs
                        st.error("❌ Ingestion finished but no retriever result found. Check logs.")
                        _log("error", "Ingestion completed but INGEST_STATE['result'] is empty.")

        # -----------------------------
        # Ask Question Section
        # -----------------------------
        st.markdown("---")
        st.subheader("💬 Ask Your Book")

        retriever_ready = (
            st.session_state.get("book_loaded")
            and st.session_state.get("retriever") is not None
        )

        if retriever_ready:
            user_q = st.text_input("Enter question:", key="question_input")
            if st.button("🔍 Get Answer") and user_q:
                with st.spinner("🤖 Thinking..."):
                    api_key = st.session_state.get("user_api_key") or os.getenv("OPENAI_API_KEY")
                    retriever = st.session_state.retriever
                    res = ask_question_fast(api_key, retriever, user_q)
                    if res.get("error"):
                        st.error(res["error"])
                    else:
                        ans = res["result"]
                        st.session_state.chat_history.append({"role": "user", "text": user_q})
                        st.session_state.chat_history.append({"role": "bot", "text": ans})
                        st.success("✅ Answer ready")
                        st.write("Answer:")
                        st.write(ans)
        else:
            st.info("📘 Upload and build index to ask questions.")

    # -----------------------------
    # Right column: Chat + Debug Info
    # -----------------------------
    with col2:
        st.subheader("📝 Conversation")
        if "chat_history" in st.session_state and st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                role = msg["role"]
                label = "You" if role == "user" else "RAG Tutor"
                st.markdown(f"{label}:** {msg['text']}")
        else:
            st.info("No chat history yet.")

        st.markdown("---")
        st.subheader("🔍 Debug Info")
        st.write("Uploaded file:", st.session_state.get("uploaded_filepath"))
        st.write("Book loaded:", st.session_state.get("book_loaded"))
        st.write("Thread running:", INGEST_STATE["in_progress"])

# -----------------------------
# Session defaults (ensures stable reruns)
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "book_loaded" not in st.session_state:
    st.session_state.book_loaded = False
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = None

main()
