# rag/ingest.py
"""
Documentation ingestion pipeline.

Fetches official Python and Pandas documentation pages, chunks them,
embeds them with sentence-transformers, and stores them in ChromaDB.

Pipeline:
    Fetch HTML → Parse text → Chunk → Embed → Store in ChromaDB
"""

import os
import sys
import time
import requests
from bs4 import BeautifulSoup


def _safe_print(msg: str):
    """Print safely on Windows cp1252 consoles by replacing unencodable chars."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from rag.vectordb import get_vectorstore, collection_exists, DEFAULT_PERSIST_DIR

# ── Documentation sources ─────────────────────────────────────────────────────
# frame.html and series.html are removed: they are huge API-signature dumps
# (~800 low-value chunks) with little explanatory text. All user-guide pages
# are kept because they contain the how-to explanations the self-healer needs.
PANDAS_DOC_URLS = [
    "https://pandas.pydata.org/docs/user_guide/indexing.html",
    "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "https://pandas.pydata.org/docs/user_guide/missing_data.html",
    "https://pandas.pydata.org/docs/user_guide/merging.html",    # joins / merge
    "https://pandas.pydata.org/docs/user_guide/reshaping.html",  # pivot / melt
    "https://pandas.pydata.org/docs/user_guide/visualization.html",
    "https://pandas.pydata.org/docs/user_guide/timeseries.html", # time series
    "https://pandas.pydata.org/docs/user_guide/basics.html",
]

PYTHON_DOC_URLS = [
    "https://docs.python.org/3/library/functions.html",
    "https://docs.python.org/3/library/stdtypes.html",   # list/dict/str methods
    "https://docs.python.org/3/library/exceptions.html", # error messages
    "https://docs.python.org/3/library/statistics.html",
]

CHUNK_SIZE    = 500   # smaller chunks → fewer total → faster embedding
CHUNK_OVERLAP = 80
HEADERS       = {"User-Agent": "Mozilla/5.0 (Data Science Copilot Doc Fetcher)"}




# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_page(url: str) -> str:
    """Download and parse text content from a documentation URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove nav, header, footer noise
        for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        main = soup.find("div", {"class": "body"}) or soup.find("main") or soup
        return main.get_text(separator="\n", strip=True)
    except Exception as exc:
        _safe_print(f"  [WARN] Could not fetch {url}: {exc}")
        return ""


def _build_documents(urls: list[str], source_tag: str) -> list[Document]:
    """Fetch pages and wrap text as LangChain Document objects."""
    docs = []
    for url in urls:
        _safe_print(f"  Fetching: {url}")
        text = _fetch_page(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url, "type": source_tag}))
        # removed crawl delay — no need to be polite for 7 pages
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ingest_documentation(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    force: bool = False,
    progress_callback=None,
) -> int:
    """
    Fetch, chunk, embed and store documentation.

    Parameters
    ----------
    persist_dir       : ChromaDB persistence directory
    force             : re-ingest even if collection already exists
    progress_callback : optional callable(message: str) for UI updates

    Returns
    -------
    Number of chunks stored.
    """

    def _log(msg: str):
        _safe_print(msg)
        if progress_callback:
            progress_callback(msg)

    if not force and collection_exists(persist_dir):
        _log("✅ Documentation already indexed — skipping ingestion.")
        return 0

    _log("📥 Fetching Pandas documentation …")
    pandas_docs = _build_documents(PANDAS_DOC_URLS, "pandas")

    _log("📥 Fetching Python documentation …")
    python_docs = _build_documents(PYTHON_DOC_URLS, "python")

    all_docs = pandas_docs + python_docs
    _log(f"📄 Total pages fetched: {len(all_docs)}")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(all_docs)
    _log(f"🔪 Total chunks created: {len(chunks)}")

    # Embed + Store
    _log("💾 Embedding and storing in ChromaDB …")
    vectorstore = get_vectorstore(persist_dir)
    # Add in batches of 500 to minimise ChromaDB roundtrips
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        _log(f"   Stored batch {i // batch_size + 1} / {(len(chunks) - 1) // batch_size + 1}")


    _log(f"✅ Ingestion complete. {len(chunks)} chunks stored in ChromaDB.")
    return len(chunks)
