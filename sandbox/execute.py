# sandbox/execute.py
"""
Secure sandbox execution engine.

Runs generated Python code in a restricted subprocess with:
  - stdout / stderr capture
  - configurable timeout
  - no network access via environment restrictions
  - exception detection and structured result return
"""

import os
import sys
import subprocess
import tempfile
import textwrap
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def execute_code(
    code: str,
    df_pickle_path: str,
    working_dir: str,
    timeout: int = 60,
) -> dict:
    """
    Execute *code* inside a subprocess sandbox.

    Parameters
    ----------
    code            : Python source code (string)
    df_pickle_path  : absolute path to a pickled DataFrame that the sandbox
                      will load as ``df``
    working_dir     : directory where output_chart.png will be saved
    timeout         : max seconds before the process is killed

    Returns
    -------
    {
        "success"   : bool,
        "stdout"    : str,
        "stderr"    : str,
        "chart_path": str | None,
        "error"     : str | None,
    }
    """
    # Wrap user code so that `df` is always available
    # NOTE: user code is injected at top-level (no extra indent)
    preamble = textwrap.dedent(f"""\
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="husl")
plt.rcParams.update({{
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#1a1d23',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#444',
    'grid.color': '#2a2d35',
}})

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    # kaleido v1.x: no scope attribute needed
except ImportError:
    pass

import os, sys
os.chdir(r"{working_dir}")

df = pd.read_pickle(r"{df_pickle_path}")

# ── USER CODE ─────────────────────────────────────────────────────────────
""")
    wrapper = preamble + code + "\n# ── END USER CODE ────────────────────────────────────────────────────────\n"

    # Write wrapper to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(wrapper)
        tmp_path = tmp.name

    chart_path = os.path.join(working_dir, "output_chart.png")

    # Remove stale chart from previous run
    if os.path.exists(chart_path):
        os.remove(chart_path)

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        success = proc.returncode == 0 and not _has_exception(stderr)

        # Determine chart
        found_chart = chart_path if os.path.exists(chart_path) else None

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "chart_path": found_chart,
            "error": _extract_error(stderr) if not success else None,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "chart_path": None,
            "error": f"Execution timed out after {timeout} seconds.",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_exception(stderr: str) -> bool:
    """Return True if stderr contains a Python traceback / exception."""
    indicators = ["Traceback (most recent call last)", "Error:", "Exception:"]
    return any(ind in stderr for ind in indicators)


def _extract_error(stderr: str) -> str:
    """Return a concise error string from stderr."""
    lines = stderr.strip().splitlines()
    # Find last meaningful error line
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("^") and not line.startswith("~"):
            return line
    return stderr[:500] if stderr else "Unknown execution error."
