# utils/token_tracker.py
"""
Token Usage Tracker — session-scoped, Groq-only.

Responsibilities:
  - Initialise token counters in st.session_state (once per session)
  - Record usage from every LLM call (code gen, repair, insights)
  - Expose totals and history for the sidebar dashboard
  - Provide a CSV export of the history

Data source: the Groq usage object from every API response.
No manual token estimation. No tiktoken.
"""

from __future__ import annotations

import io
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UsageInfo:
    """Token usage for a single LLM call."""
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0

    @classmethod
    def zero(cls) -> "UsageInfo":
        return cls(0, 0, 0)

    def __add__(self, other: "UsageInfo") -> "UsageInfo":
        return UsageInfo(
            self.prompt_tokens     + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens      + other.total_tokens,
        )

    def __bool__(self) -> bool:
        return self.total_tokens > 0


@dataclass
class UsageRecord:
    """One row in the history table."""
    timestamp:         str
    question:          str
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

_SS_PROMPT      = "tt_prompt_tokens"
_SS_COMPLETION  = "tt_completion_tokens"
_SS_TOTAL       = "tt_total_tokens"
_SS_LAST        = "tt_last_usage"        # UsageInfo for the most recent request
_SS_HISTORY     = "tt_history"           # list[UsageRecord]


def init() -> None:
    """
    Initialise all token-tracker session keys exactly once per browser session.
    Call this at the top of app.py before any other st.* call.
    """
    defaults = {
        _SS_PROMPT:     0,
        _SS_COMPLETION: 0,
        _SS_TOTAL:      0,
        _SS_LAST:       UsageInfo.zero(),
        _SS_HISTORY:    [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Record usage
# ─────────────────────────────────────────────────────────────────────────────

def record(question: str, usage: UsageInfo) -> None:
    """
    Record the token usage from one analysis run and update session totals.

    Parameters
    ----------
    question : the user's natural-language question
    usage    : UsageInfo summed across all LLM calls in this run
    """
    if usage.total_tokens == 0:
        return   # nothing to record (e.g. early failure before any LLM call)

    st.session_state[_SS_PROMPT]     += usage.prompt_tokens
    st.session_state[_SS_COMPLETION] += usage.completion_tokens
    st.session_state[_SS_TOTAL]      += usage.total_tokens
    st.session_state[_SS_LAST]        = usage

    record_obj = UsageRecord(
        timestamp         = datetime.now().strftime("%H:%M:%S"),
        question          = question[:80],
        prompt_tokens     = usage.prompt_tokens,
        completion_tokens = usage.completion_tokens,
        total_tokens      = usage.total_tokens,
    )
    st.session_state[_SS_HISTORY].append(record_obj)


# ─────────────────────────────────────────────────────────────────────────────
# Accessors
# ─────────────────────────────────────────────────────────────────────────────

def get_last() -> UsageInfo:
    return st.session_state.get(_SS_LAST, UsageInfo.zero())


def get_session_totals() -> dict:
    return {
        "prompt_tokens":     st.session_state.get(_SS_PROMPT,     0),
        "completion_tokens": st.session_state.get(_SS_COMPLETION, 0),
        "total_tokens":      st.session_state.get(_SS_TOTAL,      0),
    }


def get_history_df() -> pd.DataFrame:
    records = st.session_state.get(_SS_HISTORY, [])
    if not records:
        return pd.DataFrame(columns=["Time", "Question", "Prompt", "Completion", "Total"])
    rows = [
        {
            "Time":       r.timestamp,
            "Question":   r.question,
            "Prompt":     r.prompt_tokens,
            "Completion": r.completion_tokens,
            "Total":      r.total_tokens,
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def to_csv_bytes() -> bytes:
    """Return history as UTF-8 CSV bytes for st.download_button."""
    df = get_history_df()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def has_history() -> bool:
    return len(st.session_state.get(_SS_HISTORY, [])) > 0
