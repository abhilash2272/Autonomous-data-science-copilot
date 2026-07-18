# utils/charts.py
"""
Chart utilities for the Streamlit UI.
Renders the saved PNG chart or a Plotly fallback inside Streamlit.
"""

import os
from PIL import Image
import streamlit as st


def display_chart(chart_path: str | None) -> bool:
    """
    Display the chart saved at *chart_path* inside Streamlit.

    Returns True if a chart was found and displayed.
    """
    if chart_path and os.path.exists(chart_path):
        img = Image.open(chart_path)
        st.image(img, use_container_width=True)
        return True
    return False


def display_no_chart_placeholder():
    """Render a placeholder when no chart was produced."""
    st.info("ℹ️ No chart was generated for this query. The output may be purely textual.")
