import streamlit as st

from agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("CrewAI (single agent) + Groq `openai/gpt-oss-120b` + free DuckDuckGo search")


def get_groq_api_key() -> str:
    """
    Read the Groq API key from Streamlit secrets.

    Locally: create .streamlit/secrets.toml (see secrets.toml.example).
    On Streamlit Community Cloud: set it in your app's Settings -> Secrets.
    """
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return ""


groq_api_key = get_groq_api_key()

with st.sidebar:
    st.header("About")
    st.markdown(
        "This app uses a **single CrewAI agent** that:\n"
        "1. Searches the web with DuckDuckGo (free, no key needed)\n"
        "2. Writes a structured report using Groq's `openai/gpt-oss-120b`"
    )
    if groq_api_key:
        st.success("Groq API key loaded from secrets.")
    else:
        st.error("No GROQ_API_KEY found in Streamlit secrets.")

if not groq_api_key:
    st.warning(
        "**GROQ_API_KEY is not set.**\n\n"
        "- On Streamlit Community Cloud: go to your app's **Settings → Secrets** "
        "and add:\n```toml\nGROQ_API_KEY = \"your_real_groq_api_key\"\n```\n"
        "- Running locally: create a `.streamlit/secrets.toml` file with the "
        "same line (see `.streamlit/secrets.toml.example`)."
    )

topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. The future of solid-state batteries",
)

run_button = st.button(
    "Generate Report", type="primary", use_container_width=True, disabled=not groq_api_key
)

if run_button:
    if not topic.strip():
        st.error("Please enter a research topic.")
    else:
        with st.spinner("Researching... this can take a minute or two."):
            try:
                report = run_research(topic.strip(), groq_api_key)
                st.session_state["report"] = report
                st.session_state["topic"] = topic.strip()
            except Exception as e:
                st.error(f"Something went wrong: {e}")

if "report" in st.session_state:
    st.markdown("---")
    st.subheader(f"Report: {st.session_state['topic']}")
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
