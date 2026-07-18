# agent/agent.py
"""
Central Autonomous Data Science Agent Orchestrator.

Flow:
  1. Inspect the uploaded DataFrame.
  2. LLM → generate Python/Pandas code            (usage tracked)
  3. Execute code in secure sandbox.
  4. If fails → self-healing loop (RAG + repair)   (usage tracked per attempt)
  5. LLM → generate insights                       (usage tracked)
  6. Return structured result including cumulative UsageInfo.
"""

import os
import io
import re
import sys
import tempfile
import pickle
import pandas as pd

from agent.executor  import generate_code, generate_insights, TokenLimitError
from agent.self_heal import self_heal_loop
from sandbox.execute import execute_code
from utils.token_tracker import UsageInfo


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_summary(df: pd.DataFrame) -> tuple[str, str]:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue(), df.head(3).to_string(index=False)


def _pickle_df(df: pd.DataFrame, directory: str) -> str:
    path = os.path.join(directory, "df_sandbox.pkl")
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return path


def _validate_question(question: str, df: pd.DataFrame) -> str | None:
    """
    Light-weight relevance check. Returns an error string if the question
    references specific named entities that don't exist anywhere in the
    DataFrame, otherwise returns None (OK to proceed).
    """
    q_lower = question.lower().strip()

    # Generic / structural questions are always fine
    generic_keywords = [
        "show", "plot", "chart", "graph", "visuali", "distribution",
        "missing", "null", "duplicate", "correlation", "heatmap",
        "describe", "summary", "count", "average", "mean", "median",
        "trend", "outlier", "box", "bar", "histogram", "scatter",
        "top", "highest", "lowest", "group", "compare", "column",
        "row", "dataset", "data", "all", "each", "per", "by",
        # additional query starters that are always analytical
        "how", "what", "which", "when", "where", "who",
        "price", "value", "total", "list", "find", "get",
        "predict", "forecast", "analyse", "analyze", "calculate",
    ]
    if any(kw in q_lower for kw in generic_keywords):
        return None   # looks like a valid analytical query

    # Build a search corpus: column names + all string-column values (sample)
    col_names  = " ".join(df.columns.tolist()).lower()
    str_cols   = df.select_dtypes(include=["object", "string"]).head(200)
    col_values = " ".join(
        str_cols.values.flatten().astype(str).tolist()
    ).lower()
    corpus = col_names + " " + col_values

    # --- Entity extraction ---
    # 1. Quoted strings  →  'John Smith'  or  "Widget A"
    quoted = re.findall(r'["\']([^"\']+)["\']', question)
    # 2. Title Case multi-word  →  Boddupally Bhanu
    titled = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b', question)
    entities = list({e.strip() for e in quoted + titled if len(e.strip()) > 2})

    # 3. Lowercase bigrams of non-stopword words  →  catches "shikhar dhawan"
    _STOPS = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","shall",
        "should","may","might","must","can","could","and","or","but",
        "in","on","at","to","of","for","with","by","from","up",
        "out","off","over","under","between","among","through",
        "about","around","into","onto","how","what","when","where",
        "who","which","why","that","this","these","those",
        "i","you","he","she","it","we","they","me","him","her",
        "us","them","my","your","his","its","our","their",
        "much","many","more","most","some","any","all","both","each",
        "few","less","little","own","same","too","very","just",
        "than","then","there","here","not","no","nor","so","yet",
        "either","neither","because","since","while","if","unless",
        "until","whether","before","after","as","like","such",
        "its","also","only","tell","give","show","find","get",
    }
    _words = re.findall(r'\b[a-z]+\b', q_lower)
    for _i in range(len(_words) - 1):
        _w1, _w2 = _words[_i], _words[_i + 1]
        if (_w1 not in _STOPS and _w2 not in _STOPS
                and len(_w1) > 3 and len(_w2) > 3):
            _bg = f"{_w1} {_w2}"
            if _bg not in corpus:
                entities.append(_bg)

    if not entities:
        return None   # no specific entity detected

    missing = []
    for entity in entities:
        if entity.lower() not in corpus:
            missing.append(entity)

    if missing:
        cols_preview = ", ".join(f"'{c}'" for c in df.columns[:10])
        if len(df.columns) > 10:
            cols_preview += " ..."
        return (
            f"Cannot answer: the following {'value' if len(missing)==1 else 'values'} "
            f"{missing} {'was' if len(missing)==1 else 'were'} not found anywhere in the dataset.\n"
            f"Available columns: [{cols_preview}]\n"
            f"Please ask a question about the actual data in your file."
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    question: str,
    status_callback=None,
) -> dict:
    """
    Execute the full autonomous data-science pipeline.

    Returns
    -------
    {
        success      : bool,
        code         : str,
        stdout       : str,
        chart_path   : str | None,
        insights     : str,
        attempts     : int,
        error        : str | None,
        healed       : bool,
        token_limit  : bool,
        usage        : UsageInfo   ← cumulative across ALL LLM calls in this run
    }
    """

    def _log(stage: str, msg: str):
        _msg = f"[{stage}] {msg}"
        try:
            print(_msg)
        except UnicodeEncodeError:
            print(_msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        if status_callback:
            status_callback(stage, msg)

    total_usage = UsageInfo.zero()

    # ── 1. Inspect DataFrame ─────────────────────────────────────────────────
    df_info, sample_data = _df_summary(df)
    columns = df.columns.tolist()
    _log("INSPECT", f"Shape: {df.shape} | Columns: {columns}")

    # ── 1b. Relevance pre-check (before any LLM call) ─────────────────────────
    validation_err = _validate_question(question, df)
    if validation_err:
        _log("INSPECT", f"Query rejected: {validation_err[:80]}")
        return {
            "success": False, "code": "", "stdout": "",
            "chart_path": None, "insights": "",
            "attempts": 1, "healed": False,
            "error": validation_err, "token_limit": False,
            "usage": UsageInfo.zero(),
        }

    # ── 2. Generate code ─────────────────────────────────────────────────────
    _log("CODEGEN", "Generating Python code …")
    try:
        code, codegen_usage = generate_code(
            question=question,
            df_info=df_info,
            columns=columns,
            sample_data=sample_data,
        )
        total_usage = total_usage + codegen_usage
        _log("CODEGEN", f"Code generated ({len(code)} chars | {codegen_usage.total_tokens} tokens).")
    except TokenLimitError as tle:
        return {
            "success": False, "code": "", "stdout": "",
            "chart_path": None, "insights": "",
            "attempts": 0, "healed": False,
            "error": str(tle), "token_limit": True,
            "usage": UsageInfo.zero(),
        }

    # ── 3. Sandbox execution ─────────────────────────────────────────────────
    working_dir = tempfile.mkdtemp(prefix="ds_copilot_")
    df_pickle   = _pickle_df(df, working_dir)

    _log("EXECUTE", "Executing code in sandbox …")
    result = execute_code(code, df_pickle, working_dir)

    attempts   = 1
    healed     = False
    final_code = code

    # ── 4. Self-healing loop ──────────────────────────────────────────────────
    # Semantic / logical errors (entity not in data, column missing, etc.) cannot
    # be fixed by code repair — skip self_heal and surface the error directly.
    _SEMANTIC_PATTERNS = [
        "not found in the dataset", "was not found", "does not exist in column",
        "cannot be answered", "cannot answer", "cannot filter",
        "column not found in dataset", "not found anywhere",
        "does not appear", "valueerror",
    ]
    _err_text = (result.get("error") or result.get("stderr") or "").lower()
    _is_semantic = any(p in _err_text for p in _SEMANTIC_PATTERNS)

    if not result["success"] and not _is_semantic:
        _log("HEAL", f"Execution failed: {result.get('error')}. Starting self-healing …")
        healed = True

        def _heal_status(attempt: int, msg: str):
            _log("HEAL", f"Attempt {attempt}: {msg}")

        heal_result = self_heal_loop(
            failed_code    = code,
            error_message  = result.get("error") or result.get("stderr") or "Unknown error",
            df_info        = df_info,
            columns        = columns,
            df_pickle_path = df_pickle,
            working_dir    = working_dir,
            execute_fn     = execute_code,
            status_callback= _heal_status,
        )

        attempts     += heal_result.get("attempt", 0)
        heal_usage    = heal_result.get("usage", UsageInfo.zero())
        total_usage   = total_usage + heal_usage
        result        = heal_result
        final_code    = heal_result.get("repaired_code", code)

    elif not result["success"] and _is_semantic:
        _log("HEAL", f"Semantic error detected — skipping self-heal: {result.get('error', '')[:80]}")

    # ── 5. Generate insights ──────────────────────────────────────────────────
    insights = ""
    if result["success"]:
        _log("INSIGHTS", "Generating insights …")
        output_text = result.get("stdout", "") or "No textual output produced."
        try:
            insights, insight_usage = generate_insights(question=question, output=output_text)
            total_usage = total_usage + insight_usage
            _log("INSIGHTS", f"Insights ready ({insight_usage.total_tokens} tokens).")
        except TokenLimitError:
            insights = "⚠️ Insights unavailable — token limit reached during insight generation."

    # ── 6. Return ─────────────────────────────────────────────────────────────
    return {
        "success"    : result["success"],
        "code"       : final_code,
        "stdout"     : result.get("stdout", ""),
        "chart_path" : result.get("chart_path"),
        "insights"   : insights,
        "attempts"   : attempts,
        "error"      : result.get("error") if not result["success"] else None,
        "healed"     : healed,
        "token_limit": False,
        "usage"      : total_usage,   # ← cumulative UsageInfo for this entire run
    }
