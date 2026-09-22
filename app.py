"""
Streamlit UI for the AI Research Agent.
Renders the report as styled HTML and offers a PDF download.
PDF is generated with fpdf2 (pure Python, no system libs required).
"""

import os

import markdown as md
import streamlit as st

from agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# PDF builder -- top-level function, no indentation
# ---------------------------------------------------------------------------
def _build_pdf(markdown_text: str) -> bytes:
    """Convert the markdown report into a PDF using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    USABLE_WIDTH = pdf.epw - 10

    def _safe(text: str) -> str:
        text = (
            text.replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2022", "-")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )
        out_words = []
        for w in text.split(" "):
            while len(w) > 60:
                out_words.append(w[:60])
                w = w[60:]
            out_words.append(w)
        return " ".join(out_words)

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("|---") or (
            line.startswith("|") and set(line.replace("|", "").strip()) == {"-"}
        ):
            continue

        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.ln(2)
            pdf.multi_cell(USABLE_WIDTH, 10, _safe(line[2:]))
            pdf.ln(2)
            pdf.set_font("Helvetica", size=11)

        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(3)
            pdf.multi_cell(USABLE_WIDTH, 8, _safe(line[3:]))
            pdf.ln(1)
            pdf.set_font("Helvetica", size=11)

        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(2)
            pdf.multi_cell(USABLE_WIDTH, 7, _safe(line[4:]))
            pdf.set_font("Helvetica", size=11)

        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(USABLE_WIDTH, 6, _safe("   |   ".join(cells)))
            pdf.set_font("Helvetica", size=11)

        elif line.startswith("- ") or line.startswith("* "):
            pdf.multi_cell(USABLE_WIDTH, 6, _safe("  - " + line[2:]))

        elif line.strip() == "":
            pdf.ln(3)

        else:
            clean = _safe(line.replace("**", "").replace("*", ""))
            pdf.multi_cell(USABLE_WIDTH, 6, clean)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This app uses a single research pipeline that:\n\n"
        "1. Searches the web with DuckDuckGo + Wikipedia\n"
        "2. Writes a structured, topic-adaptive report using Groq's "
        "`openai/gpt-oss-120b`"
    )
    st.success("Groq API key loaded from secrets.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🔍 AI Research Agent")
st.caption("Groq `openai/gpt-oss-120b` + DuckDuckGo + Wikipedia")

topic = st.text_input("Enter a research topic", placeholder="e.g. The role of NLP")

if st.button("Generate Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    if not groq_api_key:
        st.error("GROQ_API_KEY not found in secrets or environment.")
        st.stop()

    with st.spinner("Researching and writing the report..."):
        try:
            report_md = run_research(topic, groq_api_key)
            st.session_state["report_md"] = report_md
            st.session_state["report_topic"] = topic
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.stop()


# ---------------------------------------------------------------------------
# Render + download
# ---------------------------------------------------------------------------
if "report_md" in st.session_state:
    report_md = st.session_state["report_md"]
    report_topic = st.session_state.get("report_topic", "report")

    st.markdown("---")

    html_body = md.markdown(
        report_md, extensions=["tables", "fenced_code", "toc"]
    )
    st.markdown(html_body, unsafe_allow_html=True)

    st.markdown("---")

    safe_name = (
        "".join(
            c if c.isalnum() or c in "-_ " else "_" for c in report_topic
        ).strip().replace(" ", "_")
        or "report"
    )

    try:
        pdf_bytes = _build_pdf(report_md)
        st.download_button(
            label="📄 Download report (PDF)",
            data=pdf_bytes,
            file_name=f"{safe_name}.pdf",
            mime="application/pdf",
        )
    except Exception as exc:
        import traceback
        st.error(f"PDF generation failed: {exc}")
        st.code(traceback.format_exc(), language="python")
        st.download_button(
            label="⬇️ Download report (.md)",
            data=report_md,
            file_name=f"{safe_name}.md",
            mime="text/markdown",
        )
