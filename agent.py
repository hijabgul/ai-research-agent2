"""
Research agent that produces a COMPLETE, topic-adaptive report.
Uses direct Groq calls only -- no CrewAI for generation, which avoids
the "Tool choice is none" bug entirely while staying within token limits.
"""

import re
import time

import litellm


# ---------------------------------------------------------------------------
# Groq call helper with retry
# ---------------------------------------------------------------------------
def _groq_call(api_key: str, system: str, user: str, max_tokens: int) -> str:
    """Single Groq call via litellm -- no tools, no CrewAI loop."""
    for attempt in range(4):
        try:
            resp = litellm.completion(
                model="groq/openai/gpt-oss-120b",
                api_key=api_key,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.5,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            msg = str(exc)
            # Retry only on rate limits, not on tool-choice errors
            if ("rate_limit_exceeded" in msg or "Rate limit" in msg) and attempt < 3:
                m = re.search(r"try again in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 5.0) if m else 20.0
                time.sleep(wait)
                continue
            raise
    return ""


# ---------------------------------------------------------------------------
# Phase 1: Search the web (no CrewAI, so no tool-calling bug)
# ---------------------------------------------------------------------------
def _gather_facts(topic: str, groq_api_key: str) -> str:
    """Search the web directly and distill into facts."""
    from tools.duckduckgo_tool import duckduckgo_search

    # Call the tool directly -- no CrewAI agent involved
    results = duckduckgo_search.run(f"{topic} overview facts")

    # Distill into bullet facts using a plain Groq call
    return _groq_call(
        groq_api_key,
        "You extract facts from search results. Return only bullet points with URLs.",
        f"""
Search results for "{topic}":
{results}

Extract 10-15 key facts as bullet points. Include source URLs in brackets.
Format: - fact one [url]
Do NOT write paragraphs. Do NOT call tools.
""",
        max_tokens=800,
    )


# ---------------------------------------------------------------------------
# Phase 2: Write the report (no tools, so no tool-choice error)
# ---------------------------------------------------------------------------
def _write_report(topic: str, facts: str, groq_api_key: str) -> str:
    """Write the report in 4 small, adaptive, non-truncating calls."""

    system = (
        "You are a thorough research report writer. Use the provided facts. "
        "Never invent sources. Always produce every requested section fully. "
        "Never stop early. Do NOT call any tools."
    )

    # Call 0: Decide report structure
    plan = _groq_call(
        groq_api_key,
        "You design report outlines. Output only the requested lines.",
        f"""
Topic: "{topic}"

Decide the BEST structure for a research report on this topic.
Pick TWO table sections that genuinely fit -- do NOT force
"Good Effects" / "Bad Effects" if they don't make sense.

Examples:
- Pros/cons topic -> "Good Effects" and "Bad Effects"
- Historical topic -> "Timeline of Key Events" and "Key Figures / Leaders"
- Comparison topic -> "Feature Comparison" and "Pros and Cons"
- Person/org topic -> "Key Facts" and "Achievements and Controversies"

Output EXACTLY these 5 lines, nothing else:
TABLE1_HEADING: <short heading>
TABLE1_COLUMNS: <col1> | <col2> | <col3>
TABLE2_HEADING: <short heading>
TABLE2_COLUMNS: <col1> | <col2> | <col3>
BODY_FOCUS: <one line describing what the detailed section should cover>
""",
        max_tokens=300,
    )

    def _grab(key, default):
        m = re.search(rf"{key}\s*:\s*(.+)", plan)
        return m.group(1).strip() if m else default

    t1_head = _grab("TABLE1_HEADING", "Key Points")
    t1_cols = _grab("TABLE1_COLUMNS", "Item | Description | Example")
    t2_head = _grab("TABLE2_HEADING", "Additional Points")
    t2_cols = _grab("TABLE2_COLUMNS", "Item | Description | Example")
    body_focus = _grab("BODY_FOCUS", f"an in-depth discussion of {topic}")

    # Call 1: Introduction + Detailed body
    body = _groq_call(
        groq_api_key,
        system,
        f"""
Topic: {topic}

Facts gathered:
{facts}

Write ONLY these two sections in Markdown, no tables:

## 1. Introduction
(2 paragraphs.)

## 4. Detailed Research Report
(600-900 words focused on: {body_focus}. Use subsections.)

Output ONLY these two sections. Do NOT call tools.
""",
        max_tokens=2000,
    )

    # Call 2: The two adaptive tables
    tables = _groq_call(
        groq_api_key,
        system,
        f"""
Topic: {topic}

Facts gathered:
{facts}

Produce EXACTLY these two Markdown tables and NOTHING else (no prose,
no intro, no headings besides the ones shown).

## 2. {t1_head}

| {t1_cols} |
|---|
| ... |

(at least 6 rows)

## 3. {t2_head}

| {t2_cols} |
|---|
| ... |

(at least 6 rows)

Do NOT call tools.
""",
        max_tokens=1500,
    )

    # Call 3: Conclusion + Sources
    ending = _groq_call(
        groq_api_key,
        system,
        f"""
Topic: {topic}

Facts gathered:
{facts}

Write ONLY these two sections:

## 5. Conclusion
(1-2 paragraphs.)

## 6. Sources
(A bullet list of the URLs from the facts above.)

Do NOT call tools.
""",
        max_tokens=800,
    )

    return (
        f"# Research Report: {topic}\n\n"
        f"{body.strip()}\n\n"
        f"{tables.strip()}\n\n"
        f"{ending.strip()}\n"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_research(topic: str, groq_api_key: str, max_retries: int = 4) -> str:
    """Research a topic and return the COMPLETE report as a string."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            facts = _gather_facts(topic, groq_api_key)
            report = _write_report(topic, facts, groq_api_key)
            return report
        except Exception as exc:
            msg = str(exc)
            if ("rate_limit_exceeded" in msg or "Rate limit" in msg) and attempt < max_retries:
                last_error = exc
                m = re.search(r"try again in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 5.0) if m else 20.0
                time.sleep(wait)
                continue
            raise

    raise last_error  # pragma: no cover
