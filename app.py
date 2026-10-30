"""
app.py - Main Streamlit Application for 'Chat With Your PDF'.

Combines PDF text extraction, overlapping chunking, local sentence embeddings,
isolated ChromaDB vector storage, and grounded Groq LLM generation into a
streamlined, portfolio-ready document question-answering web app.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

import streamlit as st
import logging
import warnings
from dotenv import load_dotenv

# Silence Streamlit source watcher warnings for optional multi-modal transformers submodules
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Load local environment files if present
load_dotenv(_CURRENT_DIR / ".env")
load_dotenv()

import pdf_utils
import rag_utils

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Chat With Your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Clean layout & typography */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1050px;
    }
    
    /* Header typography */
    .app-header {
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .app-subtitle {
        font-size: 1.0rem;
        color: #64748b;
        margin-top: 0;
    }
    
    /* Document Info Pill */
    .doc-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        color: #2563eb;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Grounding sources container */
    .source-box {
        background-color: rgba(148, 163, 184, 0.08);
        border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin-top: 0.6rem;
        font-size: 0.88rem;
    }
    .source-page-badge {
        background: #3b82f6;
        color: white;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 0.4rem;
        display: inline-block;
    }
    
    /* Guide card for empty state */
    .hero-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 51, 234, 0.05) 100%);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


def resolve_groq_api_key() -> Optional[str]:
    """Resolves Groq API key quietly from Streamlit secrets or environment."""
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"].strip():
            return st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        pass
    env_key = os.getenv("GROQ_API_KEY")
    if env_key and env_key.strip() and "your_groq_api_key_here" not in env_key:
        return env_key.strip()
    return None


def init_session_state():
    """Initializes persistent session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "doc_hash" not in st.session_state:
        st.session_state.doc_hash = None
    if "doc_info" not in st.session_state:
        st.session_state.doc_info = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def reset_session():
    """Resets chat history and clears active document context."""
    st.session_state.messages = []
    st.session_state.doc_hash = None
    st.session_state.doc_info = None
    st.session_state.uploader_key += 1
    st.rerun()


def main():
    init_session_state()

    # --- Sidebar Setup ---
    with st.sidebar:
        st.title("📄 PDF Assistant")
        st.caption("AI-Powered Document Assistant")
        st.markdown("---")

        active_api_key = resolve_groq_api_key()

        # LLM Model & Settings
        st.subheader("⚙️ Settings")
        available_models = rag_utils.get_available_groq_models(active_api_key)
        selected_model = st.selectbox(
            "Groq LLM Model",
            options=available_models,
            index=0,
            help="High-throughput, strictly grounded models on Groq."
        )

        top_k = st.slider(
            "Retrieved Passages (Top-K)",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of most relevant PDF passages supplied to the LLM."
        )

        # Document Upload in Sidebar
        st.markdown("---")
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key=f"pdf_uploader_{st.session_state.uploader_key}",
            help="Upload a text-based PDF document up to 25MB."
        )

        # New Chat / Reset Action Button
        st.markdown("---")
        if st.button("🔄 New Chat / Upload New PDF", use_container_width=True, type="secondary"):
            reset_session()

        # Instructions Accordion
        with st.expander("ℹ️ How it works", expanded=False):
            st.markdown(f"""
            1. **Upload** any text-based PDF.
            2. **Processing** extracts pages, slices overlapping chunks (250 words / 50 overlap), and embeds vectors using **`all-MiniLM-L6-v2`** into local ChromaDB.
            3. **Ask** questions in the chat.
            4. **Grounding**: Answers cite exact PDF page numbers using retrieved chunks. If the answer isn't in the PDF, the assistant admits it cannot answer.
            
            **Stack**: Streamlit · pypdf · sentence-transformers · ChromaDB · Groq LLM
            """)

    # --- Main Header ---
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Chat With Your PDF</div>
        <div class="app-subtitle">Strictly grounded question-answering with verifiable page citations powered by ChromaDB & Groq</div>
    </div>
    """, unsafe_allow_html=True)

    # --- PDF Processing Logic ---
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        incoming_hash = pdf_utils.get_pdf_hash(file_bytes)

        # Detect new document upload
        if st.session_state.doc_hash != incoming_hash:
            with st.status("Processing PDF document...", expanded=True) as status:
                try:
                    st.write("📖 Extracting text and preserving page numbers...")
                    processed = pdf_utils.process_pdf(
                        pdf_source=file_bytes,
                        source_name=uploaded_file.name
                    )

                    st.write(f"🧩 Split into {len(processed['chunks'])} overlapping chunks (250 words / 50 overlap)...")
                    st.write("🔍 Generating local sentence embeddings & indexing ChromaDB...")
                    
                    col, was_new = rag_utils.index_pdf_chunks(
                        chunks=processed["chunks"],
                        doc_hash=incoming_hash
                    )

                    # Update session state with clean document context
                    st.session_state.doc_hash = incoming_hash
                    st.session_state.doc_info = {
                        "name": uploaded_file.name,
                        "pages": processed["total_pages"],
                        "words": processed["total_words"],
                        "chunks": len(processed["chunks"])
                    }
                    # Clear previous chat messages when a new PDF is uploaded
                    st.session_state.messages = []
                    status.update(label=f"Ready! Indexed '{uploaded_file.name}'", state="complete", expanded=False)
                    st.toast(f"Successfully indexed {uploaded_file.name}", icon="🚀")

                except pdf_utils.EncryptedPDFError as e:
                    status.update(label="Failed to process PDF", state="error", expanded=True)
                    st.error(f"🔒 **Protected File**: {e}")
                    return
                except pdf_utils.NoTextExtractedError as e:
                    status.update(label="No extractable text found", state="error", expanded=True)
                    st.error(f"⚠️ **Image / Scanned PDF**: {e}")
                    return
                except pdf_utils.CorruptedPDFError as e:
                    status.update(label="Corrupted PDF", state="error", expanded=True)
                    st.error(f"❌ **Invalid PDF**: {e}")
                    return
                except Exception as e:
                    status.update(label="Unexpected Processing Error", state="error", expanded=True)
                    st.error(f"❌ Error indexing PDF: {str(e)}")
                    return

    # --- Render Active Document Status Banner ---
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown(
            f'<div class="doc-pill">📄 Active Document: <strong>{info["name"]}</strong> &nbsp;|&nbsp; '
            f'{info["pages"]} Pages &nbsp;|&nbsp; {info["words"]:,} Words &nbsp;|&nbsp; '
            f'{info["chunks"]} Chunks</div>',
            unsafe_allow_html=True
        )
    else:
        # Empty State Hero Guide
        st.markdown("""
        <div class="hero-card">
            <h3>Welcome to Chat With Your PDF</h3>
            <p style="color: #64748b; max-width: 650px; margin: 0.5rem auto 1.5rem auto;">
                Upload a PDF file using the sidebar on the left to begin asking questions.
                All responses are strictly grounded in your document's text and cite source page numbers.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; text-align: left; max-width: 600px; margin: 0 auto; font-size: 0.9rem;">
                <div>
                    <strong>🛡️ Zero Hallucinations</strong><br>
                    Declines to answer if context is absent.
                </div>
                <div>
                    <strong>📑 Page Citations</strong><br>
                    Every answer attributes source pages.
                </div>
                <div>
                    <strong>⚡ Isolated Sessions</strong><br>
                    Independent vector stores per document.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- Render Conversation History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])} passages)", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<span class="source-page-badge">{src["page_label"]}</span> '
                            f'<span style="color:#64748b; font-size:0.75rem;">(Similarity: {src["similarity"]:.2f})</span><br>'
                            f'{src["excerpt"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # --- Chat Input & Execution ---
    user_query = st.chat_input("Ask a question about the uploaded PDF...")

    if user_query:
        clean_query = user_query.strip()
        if not clean_query:
            st.warning("Please enter a question.")
            return

        if not active_api_key:
            st.error(
                "⚠️ **Configuration Error**: Groq API Key is not configured. Please ensure `GROQ_API_KEY` is set in `.env` or `.streamlit/secrets.toml`.",
                icon="⚠️"
            )
            return

        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": clean_query})
        with st.chat_message("user"):
            st.markdown(clean_query)

        # 2. Assistant Response Turn
        with st.chat_message("assistant"):
            with st.spinner("Searching document & formulating grounded answer..."):
                try:
                    result = rag_utils.generate_rag_answer(
                        question=clean_query,
                        doc_hash=st.session_state.doc_hash,
                        api_key=active_api_key,
                        top_k=top_k,
                        model_name=selected_model
                    )

                    answer_text = result["answer"]
                    sources = result["sources"]

                    # Display Answer
                    st.markdown(answer_text)

                    # Display Sources if available
                    if sources and not result["is_grounded_fallback"]:
                        with st.expander(f"📚 Retrieved Sources ({len(sources)} passages)", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<span class="source-page-badge">{src["page_label"]}</span> '
                                    f'<span style="color:#64748b; font-size:0.75rem;">(Similarity: {src["similarity"]:.2f})</span><br>'
                                    f'{src["excerpt"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                    # Save to conversation history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources if not result["is_grounded_fallback"] else []
                    })

                except Exception as e:
                    error_msg = f"❌ **Error generating response**: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })


if __name__ == "__main__":
    main()
