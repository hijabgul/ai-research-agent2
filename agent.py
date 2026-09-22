"""
General-purpose research agent.

Sources tried in parallel (any topic works):
  1. Google News RSS   -- fresh current-events coverage (no API key)
  2. DuckDuckGo        -- general web search
  3. Wikipedia REST    -- reliable structured background

Then 4 small Groq calls write the report:
  plan -> body -> tables -> conclusion
"""

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import litellm


def _groq_call(api_key: str, system: str, user: str, max_tokens: int) -> str:
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


# ---------------------------------------------------------------------------
# Clean the topic before searching
# ---------------------------------------------------------------------------
def _clean_topic(topic: str) -> str:
    """Strip instruction phrases so search gets a real query."""
    t = topic.lower().strip().rstrip("?.")

    prefixes = [
        "write a detailed report on", "write a detailed report about",
        "write a report on", "write a report about",
        "write about", "give me a report on", "give me info about",
        "tell me about", "research", "report on", "a report on",
        "detailed report on", "comprehensive report on",
        "please write", "please create", "please generate",
    ]
    for p in prefixes:
        if t.startswith(p):
            t = t[len(p):].strip()

    suffixes = [
        "write a detailed report", "write a report", "write report",
        "detailed report", "in detail", "please", "thanks",
    ]
    for s in suffixes:
        if t.endswith(s):
            t = t[:-len(s)].strip()

    return " ".join(t.split()) or topic


# ---------------------------------------------------------------------------
# Source 1: Google News RSS
# ---------------------------------------------------------------------------
def _google_news(topic: str, max_items: int = 6) -> str:
    try:
        q = urllib.parse.quote(topic)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (research-agent)"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            xml_bytes = r.read()
        root = ET.fromstring(xml_bytes)
        lines = []
        for it in root.findall(".//item")[:max_items]:
            title = (it.findtext("title") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            link = (it.findtext("link") or "").strip()
            if title:
                lines.append(f"- {title} ({pub}) [{link}]")
        return "\n".join(lines)
    except Exception as e:
        return f"[Google News failed: {e}]"


# ---------------------------------------------------------------------------
# Source 2: DuckDuckGo
# ---------------------------------------------------------------------------
def _duckduckgo(topic: str, max_items: int = 6) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=max_items))
        return "\n".join(
            f"- {r.get('title','')}: {r.get('body','')} [{r.get('href','')}]"
            for r in results
            if r.get("body")
        )
    except Exception as e:
        return f"[DDG failed: {e}]"


# ---------------------------------------------------------------------------
# Source 3: Wikipedia
# ---------------------------------------------------------------------------
def _wikipedia_summary(title: str) -> str:
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title.replace(" ", "_"))
    )
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (research-agent)"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        return data.get("extract", "").strip()
    except Exception:
        return ""


def _best_wikipedia_title(topic: str) -> str:
    t = topic.strip().rstrip("?.").lower()
    m = re.search(
        r"(prime ministers?|presidents?|kings?|chief ministers?)\s+of\s+([a-z\s]+)",
        t,
    )
    if m:
        role = m.group(1).strip().title()
        country = m.group(2).strip().title().split(" from ")[0].split(" to ")[0].strip()
        return f"List of {role.lower()} of {country}"
    cleaned = re.sub(r"\b\d{4}\b", "", t)
    cleaned = re.sub(r"\b(the|a|an|from|to|in|on|of)\b", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned.title() if cleaned else topic


# ---------------------------------------------------------------------------
# Gather facts
# ---------------------------------------------------------------------------
def _gather_facts(topic: str, groq_api_key: str) -> str:
    topic = _clean_topic(topic)

    news = _google_news(topic)
    ddg = _duckduckgo(topic)
    wiki_title = _best_wikipedia_title(topic)
    wiki = _wikipedia_summary(wiki_title)

    combined = f"""
=== GOOGLE NEWS (freshest) ===
{news}

=== DUCKDUCKGO ===
{ddg}

=== WIKIPEDIA ({wiki_title}) ===
{wiki}
""".strip()

    if len(combined) < 200:
        return _groq_call(
            groq_api_key,
            "You are a knowledge extractor. Output only bullet facts.",
            f"""Web sources for "{topic}" all failed. Use your own knowledge.

Produce 15 bullet facts about "{topic}". Each bullet: - <fact>

If the topic requires post-2023 info, add:
- NOTE: my knowledge may be outdated; verify recent facts independently.

At the very end add:
SOURCES: (model knowledge -- live sources unavailable)""",
            max_tokens=900,
        )

    return _groq_call(
        groq_api_key,
        "You extract facts from sources. Output ONLY bullet points.",
        f"""Sources about "{topic}":

{combined[:6000]}

Extract 15-20 key facts as bullets.

RULES:
1. NEWS items are the FRESHEST -- prioritize them for current events.
2. If news mentions a person holding an office TODAY, include that
   person and the date.
3. Wikipedia gives background and history -- use it for earlier facts.
4. NEVER invent. If a fact is not in the sources, don't write it.
5. Include the source URL in brackets for each bullet.
6. Order bullets chronologically where possible.

Format: - fact one [url]

At the very end add:
SOURCES: <comma-separated URLs>""",
        max_tokens=1200,
    )


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
def _write_report(topic: str, facts: str, groq_api_key: str) -> str:
    system = (
        "You are a thorough research report writer. Use the provided facts. "
        "Never invent sources. Always produce every requested section fully. "
        "Never stop early. Do NOT call any tools. "
        "Prefer the freshest facts (Google News) over your own training data."
    )

    plan = _groq_call(
        groq_api_key,
        "You design report outlines. Output only the requested lines.",
        f"""Topic: "{topic}"

Pick the BEST structure for a research report on this topic.
Choose TWO table sections that genuinely fit.

Examples:
- Pros/cons        -> "Good Effects" / "Bad Effects"
- Historical       -> "Timeline of Key Events" / "Key Figures"
- Comparison       -> "Feature Comparison" / "Pros and Cons"
- Person/org       -> "Key Facts" / "Achievements and Controversies"
- Scientific       -> "Benefits" / "Risks and Limitations"
- Country/economy  -> "Key Indicators" / "Recent Developments"

Output EXACTLY these 5 lines and nothing else:
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

    body = _groq_call(
        groq_api_key,
        system,
        f"""Topic: {topic}

Facts gathered (from Google News + DuckDuckGo + Wikipedia):
{facts}

Write ONLY these two sections in Markdown, no tables:

## 1. Introduction
(2 paragraphs. If the facts reveal a CURRENT state -- e.g. a current
office-holder, current status, latest event -- state it explicitly in
the first paragraph with the date.)

## 4. Detailed Research Report
(600-900 words focused on: {body_focus}. Use subsections.)

Do NOT invent facts that aren't in the source material.

Output ONLY these two sections. Do NOT call tools.""",
        max_tokens=1500,
    )

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
- At least 6 rows per table.

Do NOT call tools.""",
        max_tokens=1500,
    )

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
Bullet list of URLs from the facts above.

Do NOT call tools.""",
        max_tokens=800,
    )

    return (
        f"# Research Report: {topic}\n\n"
        f"{body.strip()}\n\n"
        f"{tables.strip()}\n\n"
        f"{ending.strip()}\n"
    )


def run_research(topic: str, groq_api_key: str, max_retries: int = 4) -> str:
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
