# 🧠 DocMind — Multi-Document RAG Chatbot

> **Resume line:** *Built a RAG-based Q&A system over custom document corpora using LangChain, ChromaDB, and Google Gemini; deployed on Streamlit with cited source retrieval.*

---

## 🔍 What It Does

Upload multiple PDFs → Ask questions in natural language → Get accurate, **cited** answers sourced directly from your documents.

No hallucination. Every answer references the exact document and page it came from.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                      │
│                                                                 │
│  PDF File ──► PyPDFLoader ──► RecursiveTextSplitter ──►        │
│                                 (1000 char chunks, 150 overlap) │
│              GoogleGenerativeAIEmbeddings (embedding-001) ──►   │
│                                 ChromaDB (persisted on disk)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                         │
│                                                                 │
│  User Question ──► Embed query ──► MMR Retrieval (top-5)       │
│                                        ↓                        │
│                                   Relevant Chunks               │
│                                        ↓                        │
│                  Custom Prompt + Chunks ──► Gemini 1.5 Flash   │
│                                        ↓                        │
│                              Answer + Source Citations          │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **MMR retrieval** (Maximum Marginal Relevance) — fetches diverse chunks, avoiding repetitive context
- **Chunk overlap (150 chars)** — preserves sentence context across boundaries
- **`RetrievalQAWithSourcesChain`** — natively returns source metadata
- **Constrained prompt** — LLM is instructed to answer ONLY from retrieved docs, preventing hallucination

---

## 🚀 Local Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/multi-doc-rag-chatbot.git
cd multi-doc-rag-chatbot
pip install -r requirements.txt
```

### 2. Get a free Gemini API key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create a new API key (free tier works)
3. Copy it

### 3. Set your API key

```bash
cp .env.example .env
# Open .env and paste your key
```

Or export directly:

```bash
export GOOGLE_API_KEY="AIza..."
```

### 4. Run

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, branch `main`, file `app.py`
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "AIza..."
   ```
5. Click **Deploy** — live in ~2 minutes

> **Note:** Streamlit Cloud has ephemeral storage, so ChromaDB resets on each restart. For production persistence, swap ChromaDB for a hosted vector DB like [Pinecone](https://pinecone.io) (free tier available).

---

## 📁 Project Structure

```
multi-doc-rag-chatbot/
├── app.py              # Streamlit UI — chat interface, sidebar, file upload
├── rag_pipeline.py     # Core RAG logic — load, chunk, embed, store, query
├── requirements.txt    # Python dependencies (pinned versions)
├── .env.example        # Template for local secrets
├── .gitignore
└── .streamlit/
    └── secrets.toml.example  # Template for Streamlit Cloud secrets
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| PDF Loading | LangChain + PyPDF |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | Google `embedding-001` |
| Vector DB | ChromaDB (local persist) |
| LLM | Google Gemini 1.5 Flash |
| RAG Chain | LangChain `RetrievalQAWithSourcesChain` |

---

## 🎯 JD Keywords This Project Covers

`RAG` · `LangChain` · `Vector DB` · `ChromaDB` · `Prompt Engineering` · `LLM` · `Google Gemini` · `Streamlit` · `Document AI` · `Embeddings`

---

## 🔮 Possible Extensions

- [ ] **Pinecone** — replace ChromaDB for cloud-native vector store
- [ ] **Conversational memory** — multi-turn conversation using `ConversationBufferMemory`
- [ ] **Hybrid search** — combine dense (embedding) + sparse (BM25) retrieval
- [ ] **Re-ranking** — use Cohere Rerank or a cross-encoder after retrieval
- [ ] **Multi-modal** — add image extraction from PDFs using `unstructured`
- [ ] **Evaluation** — add RAGAS metrics (faithfulness, answer relevancy, context recall)
