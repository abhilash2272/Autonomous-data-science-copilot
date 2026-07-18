# 🤖 Autonomous Data Science Co-Pilot

An AI-powered data analysis assistant built with **Groq LLM**, **LangChain**, **ChromaDB RAG**, and **Streamlit**.

Upload any CSV, Excel, or JSON dataset → ask questions in plain English → get professional charts, insights & self-healing code analysis — zero coding required.

---

## ✨ Features

- 📊 **Auto-generated charts** from natural language questions
- 🔮 **AI-generated questions** tailored to your dataset
- 🔧 **Self-healing code** — automatically fixes errors using RAG over Pandas/Python docs
- 📋 **Session history** — tracks all your analyses
- 🗄️ **ChromaDB RAG** — retrieval-augmented generation for smarter code repair

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/autonomous-data-science-copilot.git
cd autonomous-data-science-copilot
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Copy the example env file and add your Groq key:
```bash
copy .env.example .env
```
Then edit `.env`:
```
GROQ_API_KEY=your-groq-api-key-here
```
> Get a **free** Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
autonomous-data-science-copilot/
├── app.py              # Main Streamlit application
├── agent/              # LLM orchestration & code generation
│   ├── agent.py        # Pipeline: Inspect → CodeGen → Execute → Heal → Insights
│   ├── executor.py     # Groq LLM calls
│   ├── prompts.py      # Prompt templates
│   └── self_heal.py    # RAG-assisted self-healing loop
├── rag/                # ChromaDB RAG system
│   ├── embeddings.py   # HuggingFace sentence-transformers
│   ├── ingest.py       # Documentation ingestion pipeline
│   └── vectordb.py     # ChromaDB vector store
├── sandbox/            # Secure code execution
│   └── execute.py      # Subprocess-isolated execution
├── utils/              # Utilities
│   ├── charts.py       # Chart display helpers
│   ├── loader.py       # Dataset loader (CSV/Excel/JSON)
│   └── token_tracker.py# Token usage tracking
├── .env.example        # Template for environment variables
├── .gitignore          # Excludes .env, venv, chroma_db, etc.
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Configuration

All configuration is done via the `.env` file (never committed to git):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `MAX_RETRIES` | `5` | Max self-heal attempts |

---

## 🔒 Security

- **Never commit your `.env` file** — it is in `.gitignore`
- The app reads `GROQ_API_KEY` from environment only — no key is hardcoded
- If the key is missing, the app shows a setup guide instead of crashing

---

## 📦 Requirements

- Python 3.10+
- Internet connection (for Groq API calls and first-time doc indexing)
