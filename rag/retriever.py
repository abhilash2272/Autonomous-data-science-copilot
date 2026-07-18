# rag/retriever.py
"""
RAG retriever — similarity search over the ChromaDB documentation index.
Used exclusively for error-correction (self-healing loop).
"""

from rag.vectordb import get_vectorstore, DEFAULT_PERSIST_DIR

TOP_K = 5   # number of document chunks to retrieve per query


def retrieve_documentation(
    query: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    k: int = TOP_K,
) -> str:
    """
    Retrieve the most relevant documentation chunks for a given error query.

    Parameters
    ----------
    query       : the error message / failed code snippet to search against
    persist_dir : ChromaDB persistence directory
    k           : number of chunks to return

    Returns
    -------
    A single string containing all retrieved chunks separated by dividers.
    """
    vectorstore = get_vectorstore(persist_dir)
    results = vectorstore.similarity_search(query, k=k)

    if not results:
        return "No relevant documentation found."

    parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "")
        header = f"[Doc {i} | {doc_type.upper()} | {source}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(parts)
