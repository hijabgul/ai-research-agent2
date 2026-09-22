"""
Research agent that produces a COMPLETE, topic-adaptive report.

Architecture:
  Phase 1: Search the web via ddgs + Wikipedia REST API for fresh facts.
  Phase 2: Direct Groq calls write the report in 4 small, adaptive chunks.
"""

import json
import re
import time
import urllib.parse
import urllib.request

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


def _wikipedia_summary(title: str) -> str:
    """Fetch a Wikipedia page summary as plain text."""
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title.replace(" ", "_"))
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
        return data.get("extract", "")
    except Exception:
        return ""


def _gather_facts(topic: str, groq_api_key: str) -> str:
    """Gather facts from DuckDuckGo + Wikipedia, prioritizing recent data."""
    results_text = ""

    # --- 1. DuckDuckGo general search ---
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            ddg = list(ddgs.text(topic, max_results=6))
        results_text += "\n".join(
            f"- {r.get('title','')}: {r.get('body','')} [{r.get('href','')}]"
            for r in ddg
            if r.get("body")
        )
    except Exception as e:
        results_text += f"\n[DDG failed: {e}]\n"

    # --- 2. Wikipedia summary for the main subject ---
    subject = topic.split(" from ")[0].split(" of ")[-1].strip()
    wiki_text = _wikipedia_summary(subject)
    if wiki_text:
        results_text += f"\n- WIKIPEDIA ({subject}): {wiki_text}"

    # --- 3. Wikipedia "List of ..." page if applicable ---
    list_candidates = [
        "List of prime ministers of Pakistan",
        "List of presidents of Pakistan",
        "List of prime ministers of India",
        "List of presidents of the United States",
    ]
    for cand in list_candidates:
        if any(word.lower() in topic.lower() for word in cand.split()[1:4]):
            list_text = _wikipedia_summary(cand)
            if list_text:
                results_text += f"\n- WIKIPEDIA LIST ({cand}): {list_text}"
            break

    # --- 4. If search yielded almost nothing, fall back to model knowledge ---
    if not results_text.strip() or len(results_text.strip()) < 200:
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

    # --- 5. Distill everything into bullets ---
    return _groq_call(
        groq_api_key,
        "You extract facts from sources. Return only bullet points.",
        f"""Sources about "{topic}":

{results_text[:5000]}

Extract 15-20 key facts as bullets.

CRITICAL RULES:
1. PREFER these sources over your own memory.
2. Include EVERY event mentioned in 2022, 2023, 2024, 2025, 2026.
3. Note the year for each fact where mentioned.
4. Include source URLs in brackets.

Format: - fact one [url]

At the end add: SOURCES: <comma-separated URLs>""",
        max_tokens=1000,
    )


def _write_report(topic: str, facts: str, groq_api_key: str) -> str:
    """Write the report in 4 small, adaptive, non-truncating calls."""

    system = (
        "You are a thorough research report writer. Use the provided facts. "
        "Never invent sources. Always produce every requested section fully. "
        "Never stop early. Do NOT call any tools. "
        "If the facts mention recent events (2022-2026), treat them as "
        "authoritative -- do NOT substitute older information from your "
        "training data."
    )

    # ---- Call 0: Decide report structure ----
    plan = _groq_call(
        groq_api_key,
        "You design report outlines. Output only the requested lines.",
        f"""Topic: "{topic}"

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
BODY_FOCUS: <one line describing what the detailed section should cover>""",
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

    # ---- Call 1: Introduction + Detailed body ----
    body = _groq_call(
        groq_api_key,
        system,
        f"""Topic: {topic}

Facts gathered (CURRENT as of 2026 -- trust these over memory):
{facts}

Write ONLY these two sections in Markdown, no tables:

## 1. Introduction
(2 paragraphs. In the FIRST paragraph, explicitly state who currently
holds the relevant office / what is the current status as of 2026.
If the facts do not say, write "As of 2026, current information is
unavailable from the search." -- do NOT guess from memory.)

## 4. Detailed Research Report
(600-900 words focused on: {body_focus}. Use subsections.)

CRITICAL: Your training data ends around 2023. Do NOT present
2023-era information as if it were current. Where the facts and your
memory disagree, USE THE FACTS. If the facts do not cover a recent
event, say so -- do NOT fill the gap from memory.

Output ONLY these two sections. Do NOT call tools.""",
        max_tokens=1500,
    )

    # ---- Call 2: The two adaptive tables (forced to have real data) ----
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
- If you do not have enough data for a row, still fill it with what
  you know from the facts.

Do NOT call tools.""",
        max_tokens=1500,
    )

    # ---- Call 3: Conclusion + Sources ----
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
