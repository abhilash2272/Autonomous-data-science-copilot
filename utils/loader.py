# utils/loader.py
"""
Dataset loader — supports CSV, Excel (.xlsx / .xls) and JSON.
PDF and other formats are explicitly rejected with a helpful message.
Returns a clean pandas DataFrame and metadata dict.
"""

import io
import json
import pandas as pd
import streamlit as st

SUPPORTED_EXTS = {"csv", "xlsx", "xls", "json"}


def load_dataframe(uploaded_file) -> tuple:
    """
    Load a Streamlit UploadedFile into a pandas DataFrame.

    Returns
    -------
    (df, metadata)
    df       : pandas DataFrame or None on failure
    metadata : dict with file info, shape, dtypes, missing values
    """
    if uploaded_file is None:
        return None, {}

    name = uploaded_file.name
    ext  = name.rsplit(".", 1)[-1].lower()
    size = uploaded_file.size

    # ── Explicit unsupported-format gates ─────────────────────────────────────
    if ext == "pdf":
        st.error(
            "❌ **PDF files are not supported.**\n\n"
            "This app works with **structured tabular data** only.\n"
            "Please upload a **CSV**, **Excel (.xlsx)**, or **JSON** file."
        )
        return None, {}

    if ext not in SUPPORTED_EXTS:
        st.error(
            f"❌ **Unsupported format: `.{ext}`**\n\n"
            "Accepted formats: **CSV**, **Excel (.xlsx / .xls)**, **JSON**."
        )
        return None, {}

    try:
        raw = uploaded_file.read()

        if ext == "csv":
            df = _load_csv(raw)
        elif ext in ("xlsx", "xls"):
            df = _load_excel(raw)
        elif ext == "json":
            df = _load_json(raw)
        else:
            return None, {}

        # ── Basic cleaning ────────────────────────────────────────────────────
        df.columns = [str(c).strip() for c in df.columns]

        metadata = {
            "name"       : name,
            "size_kb"    : round(size / 1024, 1),
            "extension"  : ext,
            "rows"       : len(df),
            "cols"       : len(df.columns),
            "columns"    : df.columns.tolist(),
            "dtypes"     : df.dtypes.astype(str).to_dict(),
            "missing"    : df.isnull().sum().to_dict(),
            "missing_pct": {
                col: round(df[col].isnull().mean() * 100, 1)
                for col in df.columns
            },
        }
        return df, metadata

    except Exception as exc:
        st.error(f"❌ Failed to load `{name}`: {exc}")
        return None, {}


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(raw: bytes) -> pd.DataFrame:
    """Try UTF-8 → UTF-8-BOM → latin-1 encoding fallback chain."""
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    # Last resort: ignore bad bytes
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="ignore")


def _load_excel(raw: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw), engine="openpyxl")


def _load_json(raw: bytes) -> pd.DataFrame:
    """Handle JSON array, single object, and newline-delimited JSON."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try records-style dict or single record
            try:
                return pd.DataFrame(data)
            except Exception:
                return pd.DataFrame([data])
    except (json.JSONDecodeError, ValueError):
        pass
    # Newline-delimited JSON
    return pd.read_json(io.BytesIO(raw), lines=True)
