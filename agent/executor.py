# agent/executor.py
"""
LLM wrapper — Groq-powered code generation, repair and insight synthesis.

Every public function returns (result_string, UsageInfo) so the caller can
accumulate real token counts from the Groq usage object.
No manual estimation. No tiktoken.
"""

import os
from langchain_groq import ChatGroq

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from langchain.schema import HumanMessage

from agent.prompts import CODE_GENERATION_PROMPT, CODE_REPAIR_PROMPT, INSIGHTS_PROMPT
from utils.token_tracker import UsageInfo

# ─────────────────────────────────────────────────────────────────────────────
# Token / rate-limit error detection
# ─────────────────────────────────────────────────────────────────────────────
TOKEN_LIMIT_PHRASES = [
    "rate_limit_exceeded", "rate limit", "tokens per", "token limit",
    "context_length_exceeded", "context length", "maximum context",
    "Request too large", "exceeded", "quota", "429",
]


class TokenLimitError(Exception):
    """Raised when the Groq API signals a rate/token limit."""
    pass


def _is_token_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p.lower() in msg for p in TOKEN_LIMIT_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        max_tokens=4096,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Usage extraction from Groq response
# ─────────────────────────────────────────────────────────────────────────────

def _extract_usage(response) -> UsageInfo:
    """
    Extract token usage from a LangChain ChatGroq response object.

    Tries multiple locations in order of preference:
      1. response.usage_metadata            (LangChain ≥ 0.3 standard)
      2. response.response_metadata         (langchain-groq specific)
      3. Falls back to UsageInfo.zero()     (never crashes)
    """
    # ── Attempt 1: usage_metadata (langchain_core standard) ──────────────────
    try:
        um = response.usage_metadata
        if um and isinstance(um, dict):
            return UsageInfo(
                prompt_tokens     = int(um.get("input_tokens",  um.get("prompt_tokens",     0))),
                completion_tokens = int(um.get("output_tokens", um.get("completion_tokens", 0))),
                total_tokens      = int(um.get("total_tokens",  0)),
            )
    except (AttributeError, TypeError, ValueError):
        pass

    # ── Attempt 2: response_metadata → token_usage ────────────────────────────
    try:
        rm = response.response_metadata
        if rm and isinstance(rm, dict):
            tu = rm.get("token_usage", rm.get("usage", {}))
            if tu:
                prompt     = int(tu.get("prompt_tokens",     tu.get("input_tokens",  0)))
                completion = int(tu.get("completion_tokens", tu.get("output_tokens", 0)))
                total      = int(tu.get("total_tokens",      prompt + completion))
                return UsageInfo(prompt, completion, total)
    except (AttributeError, TypeError, ValueError):
        pass

    return UsageInfo.zero()


# ─────────────────────────────────────────────────────────────────────────────
# Core invoke — returns (content_str, UsageInfo)
# ─────────────────────────────────────────────────────────────────────────────

def _invoke(llm: ChatGroq, prompt: str) -> tuple[str, UsageInfo]:
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        usage    = _extract_usage(response)
        return response.content, usage
    except Exception as exc:
        if _is_token_error(exc):
            raise TokenLimitError(
                "⚠️ Groq token/rate limit reached. Please wait a moment and try again, "
                "or switch to **mixtral-8x7b-32768** (larger context window) in the sidebar."
            ) from exc
        raise


def _clean_code(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end   = -1 if lines[-1].strip() == "```" else len(lines)
        raw   = "\n".join(lines[start:end])
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API  — all return (result, UsageInfo)
# ─────────────────────────────────────────────────────────────────────────────

def generate_code(
    question: str,
    df_info: str,
    columns: list,
    sample_data: str,
) -> tuple[str, UsageInfo]:
    """Returns (python_code, usage)."""
    llm    = _get_llm()
    prompt = CODE_GENERATION_PROMPT.format(
        question=question,
        df_info=df_info,
        columns=columns,
        sample_data=sample_data,
    )
    content, usage = _invoke(llm, prompt)
    return _clean_code(content), usage


def repair_code(
    failed_code: str,
    error_message: str,
    documentation: str,
    df_info: str,
    columns: list,
) -> tuple[str, UsageInfo]:
    """Returns (repaired_python_code, usage)."""
    llm    = _get_llm()
    prompt = CODE_REPAIR_PROMPT.format(
        failed_code=failed_code,
        error_message=error_message,
        documentation=documentation,
        df_info=df_info,
        columns=columns,
    )
    content, usage = _invoke(llm, prompt)
    return _clean_code(content), usage


def generate_insights(question: str, output: str) -> tuple[str, UsageInfo]:
    """Returns (insights_text, usage)."""
    llm            = _get_llm(temperature=0.3)
    prompt         = INSIGHTS_PROMPT.format(question=question, output=output)
    content, usage = _invoke(llm, prompt)
    return content.strip(), usage


def generate_dataset_questions(
    columns: list,
    dtypes: dict,
    sample_data: str,
    n: int = 10,
) -> list[str]:
    """
    Generate n dataset-specific questions.
    Returns list of question strings (usage not tracked here — sidebar helper only).
    """
    col_info = ", ".join(f"{c} ({dtypes.get(c, 'unknown')})" for c in columns)
    prompt = f"""You are a data analyst. Given a dataset with these columns:
{col_info}

Sample data:
{sample_data}

Generate exactly {n} specific, useful analysis questions a business user would ask about this dataset.
Rules:
- Reference actual column names from the dataset
- Cover: totals, distributions, trends, comparisons, outliers, correlations, missing values
- Each question should be answerable with pandas + matplotlib/seaborn
- Return ONLY a plain numbered list like:
1. Question one
2. Question two
No extra text."""

    try:
        llm             = _get_llm(temperature=0.4)
        content, _usage = _invoke(llm, prompt)
        lines = [
            l.strip().lstrip("0123456789.-) ").strip()
            for l in content.strip().splitlines()
            if l.strip() and l.strip()[0].isdigit()
        ]
        return lines[:n] if lines else []
    except Exception as e:
        import traceback
        print("GENERATE QUESTIONS ERROR:")
        traceback.print_exc()
        return []
