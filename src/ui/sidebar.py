import streamlit as st

from vietnamese_ocr_advanced import VietnameseOCRAdvanced


PDF_PREFERRED_ENGINES = ["paddleocr", "tesseract", "easyocr"]


@st.cache_resource
def load_ocr_system(device: str = "cpu"):
    """Load only PDF-relevant OCR engines."""
    with st.spinner("Initializing PDF OCR engines..."):
        return VietnameseOCRAdvanced(
            device=device,
            enable_all=False,
            preferred_engines=PDF_PREFERRED_ENGINES,
        )


def render_sidebar():
    """Render the sidebar and return settings."""
    with st.sidebar:
        st.header("Settings")

        device = st.selectbox(
            "Device",
            ["cpu", "cuda"],
            help="Use CUDA only if your OCR libraries are installed with GPU support.",
        )

        st.divider()

        if (
            "ocr_system" not in st.session_state
            or st.session_state.ocr_system is None
            or st.session_state.get("ocr_device") != device
        ):
            st.session_state.ocr_system = load_ocr_system(device)
            st.session_state.ocr_device = device

        ocr_system = st.session_state.ocr_system
        available_engines = list(ocr_system.engines.keys()) if ocr_system else []

        st.markdown("### OCR Engines")
        if available_engines:
            st.caption(f"Available: {', '.join(available_engines)}")
        else:
            st.warning("No OCR engines are available yet.")

        selected_engines = []

        if "paddleocr" in available_engines:
            if st.checkbox(
                "PaddleOCR (primary for Vietnamese PDF)",
                value=True,
                key="cb_paddleocr",
            ):
                selected_engines.append("paddleocr")
        else:
            st.checkbox(
                "PaddleOCR (not installed)",
                value=False,
                disabled=True,
                key="cb_paddleocr_disabled",
                help="Install both paddlepaddle and paddleocr.",
            )

        if "tesseract" in available_engines:
            if st.checkbox(
                "Tesseract (backup for clean printed pages)",
                value=True,
                key="cb_tesseract",
            ):
                selected_engines.append("tesseract")
        else:
            st.checkbox(
                "Tesseract (not installed)",
                value=False,
                disabled=True,
                key="cb_tesseract_disabled",
                help="Install Tesseract system-wide, then keep pytesseract in Python.",
            )

        if "easyocr" in available_engines:
            if st.checkbox(
                "EasyOCR (optional fallback)",
                value=False,
                key="cb_easyocr",
            ):
                selected_engines.append("easyocr")
        else:
            st.checkbox(
                "EasyOCR (not installed)",
                value=False,
                disabled=True,
                key="cb_easyocr_disabled",
                help="Optional. Leave it out if you want a lighter setup.",
            )

        if not selected_engines and available_engines:
            st.warning("At least one OCR engine must be selected.")
            selected_engines = available_engines[:1]

        st.divider()

        voting_method = st.selectbox(
            "Voting Method",
            ["weighted", "best", "majority"],
            help="How to combine results if more than one OCR engine is active.",
        )

        st.divider()

        st.markdown("### Speed Settings")
        fast_mode = st.checkbox(
            "Fast Mode (prefer PaddleOCR only)",
            value=True,
            help="Best default for PDF conversion. Use one fast engine first, then fallback only if needed.",
        )

        st.divider()

        st.markdown("### Preprocessing")
        preprocess_enabled = st.checkbox("Enable advanced preprocessing", value=True)
        aggressive_mode = st.checkbox(
            "Aggressive enhancement",
            value=False,
            help="Use only for noisy scans or low-contrast pages.",
        )
        image_type = st.selectbox(
            "Image Type",
            ["auto", "handwritten", "printed"],
            help="For PDF pages, printed is usually the right choice.",
        )

        st.divider()

        if "processing_history" in st.session_state and st.session_state.processing_history:
            st.markdown("### Statistics")
            st.metric("Total Processed", len(st.session_state.processing_history))

            if st.button("Clear History"):
                st.session_state.processing_history = []
                st.rerun()

        st.divider()

        st.markdown("### About")
        st.markdown(
            """
        **Version:** PDF-first

        **Goal:**
        - Direct text extraction first
        - OCR only for scanned pages or image blocks
        - Editable `.docx` output
        - Better Vietnamese PDF handling

        **Recommended engine order:**
        - PaddleOCR
        - Tesseract
        - EasyOCR
        """
        )

    available_selected = [engine for engine in selected_engines if engine in available_engines]
    if not available_selected and selected_engines:
        st.error("None of the selected engines are available.")
        return None

    if len(available_selected) < len(selected_engines):
        selected_engines = available_selected

    return {
        "ocr_system": ocr_system,
        "selected_engines": selected_engines,
        "voting_method": voting_method,
        "preprocess_enabled": preprocess_enabled,
        "aggressive_mode": aggressive_mode,
        "image_type": image_type,
        "fast_mode": fast_mode,
    }
