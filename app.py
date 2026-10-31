"""
app.py — Multi-Document RAG Chatbot  (Streamlit UI)
=====================================================
Upload PDFs → ask questions → get cited answers from your documents.
API key is loaded automatically from .env — no manual entry needed.
"""

import os

# ⚠️  MUST be set BEFORE any other import.
# Fixes "TypeError: Descriptors cannot be created directly" crash on Streamlit Cloud
# caused by a protobuf version conflict with ChromaDB's OpenTelemetry dependency.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_pdf,
    chunk_documents,
    get_or_create_vectorstore,
    add_documents_to_store,
    get_indexed_docs,
    delete_document,
    query_rag,
    file_hash,
)

# ─────────────────────────────────────────────────────────────
# Load API key from .env automatically
# ─────────────────────────────────────────────────────────────

load_dotenv()
ENV_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocMind — RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — dark sidebar, chat bubbles, citation cards
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-family: 'DM Serif Display', serif !important;
    }

    .user-bubble {
        background: #1e40af; color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto; max-width: 75%;
        font-size: 15px; line-height: 1.6;
        box-shadow: 0 2px 8px rgba(30,64,175,0.3);
    }
    .assistant-bubble {
        background: white; color: #1e293b;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0; max-width: 85%;
        font-size: 15px; line-height: 1.7;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .citation-card {
        background: #f1f5f9;
        border-left: 3px solid #3b82f6;
        padding: 10px 14px; margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-size: 13px; color: #475569;
    }
    .citation-card strong { color: #1e40af; }
    .app-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem; color: #0f172a; margin-bottom: 4px;
    }
    .app-subtitle { color: #64748b; font-size: 15px; margin-bottom: 24px; }
    .stTextInput > div > div > input {
        border-radius: 24px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 12px 20px !important; font-size: 15px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

if "messages"         not in st.session_state: st.session_state.messages         = []
if "vectorstore"      not in st.session_state: st.session_state.vectorstore      = None
if "indexed_hashes"   not in st.session_state: st.session_state.indexed_hashes   = set()
if "api_key"          not in st.session_state: st.session_state.api_key          = ENV_API_KEY
if "api_key_verified" not in st.session_state:
    st.session_state.api_key_verified = bool(ENV_API_KEY)
    if ENV_API_KEY:
        try:
            st.session_state.vectorstore = get_or_create_vectorstore(ENV_API_KEY)
        except Exception:
            st.session_state.api_key_verified = False


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("*RAG Chatbot · Powered by Groq + Llama 3*")
    st.divider()

    # ── API Key ──────────────────────────────────────────────
    st.markdown("### 🔑 Groq API Key")
    if ENV_API_KEY:
        st.success("✅ Loaded automatically from .env")
    else:
        st.warning("⚠️ No .env key found — enter manually below")
        api_key_input = st.text_input(
            "Paste your Groq key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com/keys",
        )
        if api_key_input and not st.session_state.api_key_verified:
            try:
                st.session_state.vectorstore      = get_or_create_vectorstore(api_key_input)
                st.session_state.api_key_verified = True
                st.session_state.api_key          = api_key_input
                st.success("✅ API key accepted!")
            except Exception as e:
                st.error(f"Error: {e}")
        st.caption("🔗 [Get a free API key →](https://console.groq.com/keys)")

    st.divider()

    # ── PDF Upload ───────────────────────────────────────────
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files and st.session_state.api_key_verified:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            fhash = file_hash(tmp_path)
            if fhash not in st.session_state.indexed_hashes:
                with st.spinner(f"Indexing **{uploaded_file.name}**…"):
                    try:
                        pages  = load_pdf(tmp_path)
                        chunks = chunk_documents(pages)
                        n      = add_documents_to_store(
                            st.session_state.vectorstore, chunks, uploaded_file.name
                        )
                        st.session_state.indexed_hashes.add(fhash)
                        st.success(f"✅ {uploaded_file.name} — {n} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ Error indexing {uploaded_file.name}: {e}")
            else:
                st.info(f"Already indexed: {uploaded_file.name}")

            os.unlink(tmp_path)

    elif uploaded_files and not st.session_state.api_key_verified:
        st.warning("⚠️ Add your API key first (see above).")

    # ── Indexed docs list ────────────────────────────────────
    if st.session_state.vectorstore:
        indexed = get_indexed_docs(st.session_state.vectorstore)
        if indexed:
            st.divider()
            st.markdown("### 📚 Indexed Documents")
            for doc in indexed:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📄 `{doc}`")
                with col2:
                    if st.button("🗑", key=f"del_{doc}"):
                        delete_document(st.session_state.vectorstore, doc)
                        st.rerun()

    st.divider()
    if st.button("🗑 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    **💡 Tips**
    - Upload multiple PDFs at once
    - Ask cross-document questions
    - Expand "sources" under each answer to see exact pages
    """)


# ─────────────────────────────────────────────────────────────
# Main area — header
# ─────────────────────────────────────────────────────────────

st.markdown('<p class="app-header">🧠 DocMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Ask questions across your documents — cited answers, zero hallucination.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.api_key_verified:
    st.info("👈  **Step 1:** Add your Groq API key to the `.env` file, then restart the app.")
elif not st.session_state.vectorstore or not get_indexed_docs(st.session_state.vectorstore):
    st.info("👈  **Step 2:** Upload one or more PDFs in the sidebar to get started.")

# ─────────────────────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s) used", expanded=False):
                for i, cite in enumerate(msg["citations"], 1):
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<strong>[{i}] {cite["source"]} · Page {cite["page"]}</strong>'
                        f'<br>{cite["snippet"]}</div>',
                        unsafe_allow_html=True,
                    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────────────────────

with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([9, 1])
    with cols[0]:
        user_input = st.text_input(
            "question",
            placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
        )
    with cols[1]:
        submit = st.form_submit_button("Send →", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Handle query
# ─────────────────────────────────────────────────────────────

if submit and user_input.strip():
    # Duplicate submission guard — prevents wasting quota on accidental re-sends
    last_user_msg = next(
        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        None,
    )
    if user_input.strip() == last_user_msg:
        st.warning("You already asked this — see the answer above. (Duplicate blocked to save quota.)")
    elif not st.session_state.api_key_verified:
        st.error("API key not configured. Add it to your `.env` file.")
    elif not st.session_state.vectorstore or not get_indexed_docs(st.session_state.vectorstore):
        st.error("Upload at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Searching documents and generating answer…"):
            try:
                answer, citations = query_rag(
                    st.session_state.vectorstore,
                    st.session_state.api_key,
                    user_input,
                )
            except Exception as e:
                answer    = f"⚠️ Unexpected error: {e}"
                citations = []

        st.session_state.messages.append({
            "role":      "assistant",
            "content":   answer,
            "citations": citations,
        })
        st.rerun()