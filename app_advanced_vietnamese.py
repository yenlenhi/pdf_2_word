"""
Main Streamlit entrypoint.
"""

import sys
import warnings
from pathlib import Path

import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ui.sidebar import render_sidebar
from ui.styles import apply_custom_styles
from ui.tabs.pdf_ocr import render_pdf_ocr_tab


st.set_page_config(
    page_title="Vietnamese PDF to Word OCR",
    page_icon="VN",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()


def main():
    st.title("Vietnamese PDF to Word OCR")
    st.markdown(
        """
        **Focused workflow for Vietnamese PDF -> Word conversion**

        - Direct extraction first for digital PDFs
        - OCR fallback only for scanned pages
        - Structured table extraction when available
        - Editable `.docx` output instead of raw OCR dumps
        """
    )
    st.divider()

    settings = render_sidebar()
    if settings is None:
        return

    st.info(
        f"Using OCR engines: {', '.join(settings['selected_engines'])} | Voting: {settings['voting_method']}"
    )
    st.divider()

    render_pdf_ocr_tab(settings)

    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: gray; padding: 2rem;">
            <p><strong>Vietnamese PDF to Word OCR</strong></p>
            <p>Digital PDFs use direct extraction. Scanned pages fall back to focused OCR.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
