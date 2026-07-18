"""
ingest_cli.py — Command-line RAG knowledge base builder.

Run this ONCE before launching the app to pre-index the documentation:

    python ingest_cli.py

Or to force a full rebuild:

    python ingest_cli.py --force
"""

import sys
import os
import argparse

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from rag.ingest import ingest_documentation
from rag.vectordb import collection_exists, DEFAULT_PERSIST_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Build the RAG knowledge base from official Python & Pandas documentation."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force a full rebuild even if the knowledge base already exists."
    )
    parser.add_argument(
        "--persist-dir", default=DEFAULT_PERSIST_DIR,
        help=f"ChromaDB persistence directory (default: {DEFAULT_PERSIST_DIR})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Autonomous Data Science Co-Pilot — RAG Ingestion CLI")
    print("=" * 60)

    if not args.force and collection_exists(args.persist_dir):
        print("\n✅ Documentation knowledge base already exists.")
        print(f"   Location: {os.path.abspath(args.persist_dir)}")
        print("\n   Use --force to rebuild from scratch.")
        return

    print(f"\n📂 Persist directory : {os.path.abspath(args.persist_dir)}")
    print("📡 Fetching official Python & Pandas documentation …\n")

    chunks = ingest_documentation(
        persist_dir=args.persist_dir,
        force=args.force,
        progress_callback=lambda msg: print(f"  {msg}"),
    )

    print(f"\n✅ Done! {chunks} documentation chunks indexed into ChromaDB.")
    print(f"   You can now launch the app:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
