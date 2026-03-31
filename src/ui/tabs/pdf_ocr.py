import io
import tempfile
from collections import Counter

import streamlit as st
from PIL import Image

from image_preprocessing import get_preprocessing_options
from pdf_to_word import DOCX_AVAILABLE, PYMUPDF_AVAILABLE, PDFToWordConverter
from table_formatter import TableFormatter

try:
    from vietnamese_spell_checker import post_process_ocr_text
except ImportError:
    post_process_ocr_text = None


def render_pdf_ocr_tab(settings):
    st.header("PDF to Word Converter")
    st.caption(
        "Optimized for Vietnamese PDFs: direct text extraction first, OCR only for image-only pages."
    )

    pdf_file = st.file_uploader("Select PDF", type=["pdf"])
    if pdf_file is None:
        st.info("Upload a PDF to convert it into an editable Word document.")
        return

    st.success(f"Loaded: {pdf_file.name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        dpi = st.slider(
            "OCR DPI",
            min_value=150,
            max_value=350,
            value=220,
            step=10,
            help="Only used for pages that require OCR. Lower values are faster.",
        )
    with col2:
        extract_tables = st.checkbox(
            "Extract tables",
            value=True,
            help="Use PyMuPDF table extraction for digital PDFs.",
        )
    with col3:
        preprocessing_options = get_preprocessing_options()
        preprocess_key = st.selectbox(
            "OCR preprocessing",
            options=list(preprocessing_options.keys()),
            format_func=lambda key: preprocessing_options[key]["name"],
            index=max(0, list(preprocessing_options.keys()).index("medium"))
            if "medium" in preprocessing_options
            else 0,
        )

    if not DOCX_AVAILABLE:
        st.error("python-docx is missing. Install dependencies before using PDF -> Word.")
        return
    if not PYMUPDF_AVAILABLE:
        st.error("PyMuPDF is missing. Install dependencies before using PDF -> Word.")
        return

    if st.button("Convert PDF to Word", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            pdf_bytes = pdf_file.getvalue()
            pdf_engines = [
                engine
                for engine in settings["selected_engines"]
                if engine in {"paddleocr", "tesseract", "easyocr"}
            ]
            if not pdf_engines:
                pdf_engines = settings["selected_engines"]

            if settings.get("fast_mode", False):
                for preferred in ("paddleocr", "tesseract", "easyocr"):
                    if preferred in pdf_engines:
                        pdf_engines = [preferred]
                        break

            def ocr_page(image: Image.Image) -> str:
                result = settings["ocr_system"].recognize(
                    image,
                    engines=pdf_engines,
                    voting_method="best",
                    preprocess=preprocess_key,
                    fast_mode=settings.get("fast_mode", False),
                    verbose=False,
                )
                text = result.text or ""
                if text and post_process_ocr_text is not None:
                    try:
                        text = post_process_ocr_text(text, verbose=False)
                    except TypeError:
                        text = post_process_ocr_text(text)
                    except Exception:
                        pass
                return text

            def on_progress(value: float, message: str) -> None:
                progress.progress(min(max(value, 0.0), 1.0))
                status.text(message)

            converter = PDFToWordConverter(ocr_callback=ocr_page)
            output_path = tempfile.mktemp(suffix=".docx")
            result = converter.convert(
                pdf_input=io.BytesIO(pdf_bytes),
                output_path=output_path,
                dpi=dpi,
                extract_tables=extract_tables,
                progress_callback=on_progress,
            )

            if not result.success:
                progress.empty()
                status.empty()
                st.error(result.error_message or "Conversion failed.")
                return

            progress.progress(1.0)
            status.text("Conversion complete.")

            extracted_text = result.extracted_text
            word_bytes = result.docx_bytes
            st.success(f"Converted {result.pages_converted} pages.")
            st.caption(f"OCR fallback engines used for PDF pages: {', '.join(pdf_engines)}")
            strategy_counts = Counter(page.strategy for page in result.pages)
            st.caption(
                "Page strategies: "
                + ", ".join(f"{name}={count}" for name, count in sorted(strategy_counts.items()))
            )

            metric1, metric2, metric3, metric4 = st.columns(4)
            with metric1:
                st.metric("Pages", result.pages_converted)
            with metric2:
                st.metric("Tables", result.table_count)
            with metric3:
                st.metric("OCR pages", result.ocr_pages)
            with metric4:
                st.metric("Word size", f"{len(word_bytes) / 1024:.1f} KB")

            preview_tab, tables_tab, raw_tab = st.tabs(
                ["Structured Text", "Tables", "Raw Text"]
            )

            with preview_tab:
                st.text_area(
                    "Structured extraction",
                    extracted_text,
                    height=520,
                    disabled=True,
                )

            with tables_tab:
                formatter = TableFormatter()
                table_found = False
                for page in result.pages:
                    if not page.tables:
                        continue
                    table_found = True
                    st.markdown(f"### Page {page.page_num}")
                    for idx, table in enumerate(page.tables, start=1):
                        st.markdown(f"**Table {idx}**")
                        html = formatter.format_table_as_html(
                            table.rows,
                            table_title=f"Page {page.page_num} Table {idx}",
                            with_borders=True,
                            zebra_striping=True,
                        )
                        st.markdown(html, unsafe_allow_html=True)
                        st.caption(table.source)
                if not table_found:
                    st.info("No structured tables were extracted from this PDF.")

            with raw_tab:
                st.code(extracted_text or "", language="text")

            st.download_button(
                "Download Word (.docx)",
                word_bytes,
                file_name=f"{pdf_file.name[:-4]}_converted.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
            st.download_button(
                "Download text (.txt)",
                extracted_text,
                file_name=f"{pdf_file.name[:-4]}_text.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.session_state.last_pdf_result = {
                "text": extracted_text,
                "pages": result.pages_converted,
                "word_bytes": word_bytes,
                "filename": pdf_file.name,
                "ocr_pages": result.ocr_pages,
                "table_count": result.table_count,
            }
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(str(exc))
            with st.expander("Traceback"):
                import traceback

                st.code(traceback.format_exc())
