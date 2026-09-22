"""
Research agent that produces a COMPLETE, topic-adaptive report.

Architecture:
  Phase 1: Search the web via ddgs (no CrewAI -> no tool-call bug).
  Phase 2: Direct Groq calls write the report in 4 small, adaptive chunks.
"""

import re
import time

import litellm


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
                temperature=0.4,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            msg = str(exc)
            if ("rate_limit_exceeded" in msg or "Rate limit" in msg) and attempt < 3:
                m = re.search(r"try again in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 5.0) if m else 20.0
                time.sleep(wait)
                continue
            raise
    return ""


def _gather_facts(topic: str, groq_api_key: str) -> str:
    """Search the web via ddgs and distill into bullets."""
    raw = ""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{topic} facts", max_results=6))
        raw = "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')} [{r.get('href', '')}]"
            for r in results
            if r.get("body")
        )
    except Exception as exc:
        raw = f"SEARCH_FAILED: {exc}"

    if not raw or "SEARCH_FAILED" in raw or len(raw.strip()) < 100:
        return _groq_call(
            groq_api_key,
            "You are a knowledge extractor. Output only bullet facts.",
            f"""Web search for "{topic}" failed. Use your own knowledge.

Produce 12-15 bullet facts about "{topic}". Each bullet:
- <fact>

At the end add this exact line:
SOURCES: (model knowledge -- web search unavailable)""",
            max_tokens=800,
        )

    return _groq_call(
        groq_api_key,
        "You extract facts from search results. Return only bullet points.",
        f"""Search results for "{topic}":
{raw[:3000]}

Extract 12-15 key facts as bullets. Include any URLs you see in brackets.
Format: - fact one [url]

At the end add a line:
SOURCES: <comma-separated URLs found>""",
        max_tokens=900,
    )


def _write_report(topic: str, facts: str, groq_api_key: str) -> str:
    """Write the report in 4 small, adaptive, non-truncating calls."""

    system = (
        "You are a thorough research report writer. Use the provided facts. "
        "Never invent sources. Always produce every requested section fully. "
        "Never stop early. Do NOT call any tools."
    )

    # Call 0: Decide adaptive structure
    plan = _groq_call(
        groq_api_key,
        "You design report outlines. Output only the requested lines.",
        f"""Topic: "{topic}"

Decide the BEST structure for a research report on this topic.
Pick TWO table sections that genuinely fit.

Examples:
- Pros/cons topic  -> "Good Effects" / "Bad Effects"
- Historical topic -> "Timeline of Key Events" / "Key Figures"
- Comparison topic -> "Feature Comparison" / "Pros and Cons"
- Person/org topic -> "Key Facts" / "Achievements and Controversies"
- Scientific topic -> "Benefits" / "Risks and Limitations"

Output EXACTLY these 5 lines, nothing else:
TABLE1_HEADING: <short heading>
TABLE1_COLUMNS: <col1> | <col2> | <col3>
TABLE2_HEADING: <short heading>
TABLE2_COLUMNS: <col1> | <col2> | <col3>
BODY_FOCUS: <one line>""",
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
        f"""Topic: {topic}

Facts gathered:
{facts}

Write ONLY these two sections in Markdown, no tables:

## 1. Introduction
(2 paragraphs.)

## 4. Detailed Research Report
(600-900 words focused on: {body_focus}. Use subsections.)

Output ONLY these two sections. Do NOT call tools.""",
        max_tokens=1500,
    )

    # Call 2: The two adaptive tables
    tables = _groq_call(
        groq_api_key,
        system,
        f"""Topic: {topic}

Facts gathered:
{facts}

Produce EXACTLY these two Markdown tables and NOTHING else.

## 2. {t1_head}

| {t1_cols} |
|---|---|
| row 1 | ... | ... |
| row 2 | ... | ... |
| row 3 | ... | ... |
| row 4 | ... | ... |
| row 5 | ... | ... |
| row 6 | ... | ... |

## 3. {t2_head}

| {t2_cols} |
|---|---|
| row 1 | ... | ... |
| row 2 | ... | ... |
| row 3 | ... | ... |
| row 4 | ... | ... |
| row 5 | ... | ... |
| row 6 | ... | ... |

RULES:
- Every cell must contain REAL data from the facts above.
- NEVER write "...", "N/A", "Example", or leave cells blank.

Do NOT call tools.""",
        max_tokens=1500,
    )

    # Call 3: Conclusion + Sources
    ending = _groq_call(
        groq_api_key,
        system,
        f"""Topic: {topic}

Facts gathered:
{facts}

Write ONLY these two sections:

## 5. Conclusion
(1-2 paragraphs.)

## 6. Sources
If real URLs appear in the facts above, list them as bullets.
If NOT, write exactly: "Model knowledge -- live web sources unavailable."
Do NOT call tools.""",
        max_tokens=700,
    )

    return (
        f"# Research Report: {topic}\n\n"
        f"{body.strip()}\n\n"
        f"{tables.strip()}\n\n"
        f"{ending.strip()}\n"
    )


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
            if (
                ("rate_limit_exceeded" in msg or "Rate limit" in msg)
                and attempt < max_retries
            ):
                last_error = exc
                m = re.search(r"try again in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 5.0) if m else 20.0
                time.sleep(wait)
                continue
            raise

    raise last_error  # pragma: no cover
