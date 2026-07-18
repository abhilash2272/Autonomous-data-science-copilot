# agent/self_heal.py
"""
Self-healing error-correction loop.

Each repair iteration:
  1. Builds a RAG query from the error + failed code.
  2. Retrieves relevant docs from ChromaDB.
  3. Calls repair_code() (returns (code, UsageInfo)).
  4. Re-executes the repaired code.
  5. Accumulates token usage across all repair attempts.
  6. Repeats up to MAX_RETRIES times.
"""

import os
import sys
from rag.retriever import retrieve_documentation
from agent.executor import repair_code
from utils.token_tracker import UsageInfo

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))


def self_heal_loop(
    failed_code: str,
    error_message: str,
    df_info: str,
    columns: list,
    df_pickle_path: str,
    working_dir: str,
    execute_fn,
    status_callback=None,
) -> dict:
    """
    Run the self-healing loop until execution succeeds or MAX_RETRIES is reached.

    Returns
    -------
    dict with keys:
        success, stdout, stderr, chart_path, error,
        repaired_code, attempt,
        usage (UsageInfo — sum of all repair calls)
    """

    def _log(attempt: int, msg: str):
        _m = f"  [Heal attempt {attempt}] {msg}"
        try:
            print(_m)
        except UnicodeEncodeError:
            print(_m.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        if status_callback:
            status_callback(attempt, msg)

    current_code  = failed_code
    current_error = error_message
    last_result   = {}
    cumulative_usage = UsageInfo.zero()

    for attempt in range(1, MAX_RETRIES + 1):

        # ── RAG retrieval ──────────────────────────────────────────────────────
        rag_query = f"{current_error}\n\n{current_code[:500]}"
        _log(attempt, f"🔍 Retrieving docs for: {current_error[:100]}")
        documentation = retrieve_documentation(rag_query)

        # ── LLM repair ────────────────────────────────────────────────────────
        _log(attempt, "📚 Docs retrieved — asking LLM to repair code …")
        repaired_code, repair_usage = repair_code(
            failed_code   = current_code,
            error_message = current_error,
            documentation = documentation,
            df_info       = df_info,
            columns       = columns,
        )
        cumulative_usage = cumulative_usage + repair_usage
        _log(attempt, f"🔧 Repaired ({repair_usage.total_tokens} tokens). Re-executing …")

        # ── Execute repaired code ─────────────────────────────────────────────
        result = execute_fn(repaired_code, df_pickle_path, working_dir)
        last_result = result
        last_result["repaired_code"] = repaired_code
        last_result["attempt"]       = attempt
        last_result["usage"]         = cumulative_usage   # running total

        if result["success"]:
            _log(attempt, "✅ Execution succeeded after self-healing!")
            return last_result

        current_code  = repaired_code
        current_error = result.get("error") or result.get("stderr") or current_error
        _log(attempt, f"❌ Still failing: {current_error[:100]}")

    _log(MAX_RETRIES, "🛑 Max retries reached.")
    last_result.setdefault("usage", cumulative_usage)
    return last_result
