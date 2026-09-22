"""
Research agent that produces a COMPLETE, topic-adaptive report.

Architecture:
  Phase 1: CrewAI agent searches the web and returns raw facts.
  Phase 2: Direct Groq calls (no tools) write the report in 4 small chunks.
  Phase 3: Report structure adapts to the topic (no forced Good/Bad tables).

Why the split?
CrewAI + Groq's gpt-oss-120b has a known bug where tool calls fail with
"Tool choice is none, but model called a tool". By keeping tool-use
confined to the search phase, we bypass the bug entirely.
"""

import re
import time

import litellm

from crewai import Agent, Task, Crew, Process, LLM

from tools.duckduckgo_tool import duckduckgo_search


def _patch_litellm_strip_cache_breakpoint() -> None:
    """Work around the CrewAI `cache_breakpoint` bug with LiteLLM/Groq."""
    if getattr(litellm, "_cache_breakpoint_patch_applied", False):
        return

    def _clean(messages):
        for m in messages or []:
            if isinstance(m, dict):
                m.pop("cache_breakpoint", None)
        return messages

    _original_completion = litellm.completion
    _original_acompletion = litellm.acompletion

    def _patched_completion(*args, **kwargs):
        _clean(kwargs.get("messages"))
        return _original_completion(*args, **kwargs)

    async def _patched_acompletion(*args, **kwargs):
        _clean(kwargs.get("messages"))
        return await _original_acompletion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
    litellm._cache_breakpoint_patch_applied = True


_patch_litellm_strip_cache_breakpoint()


def _gather_facts(topic: str, groq_api_key: str) -> str:
    """Phase 1: CrewAI agent searches the web and returns raw facts."""
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.3,
        max_tokens=800,
    )

    researcher = Agent(
        role="Research Analyst",
        goal=f"Search the web for facts about '{topic}'.",
        backstory="Concise research analyst. Search first, return bullet facts with URLs.",
        tools=[duckduckgo_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    task = Task(
        description=(
            f"Use the DuckDuckGo Search tool to research: '{topic}'.\n\n"
            "Return ONLY a compact bullet list of facts and source URLs. "
            "Do NOT write a report. Do NOT write paragraphs. "
            "Format:\n"
            "- fact one [url]\n"
            "- fact two [url]\n"
            "- fact three [url]\n"
            "...\n"
            "10-15 bullets maximum."
        ),
        expected_output="A short bullet list of facts with URLs. No prose.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


def _groq_call(api_key: str, system: str, user: str, max_tokens: int) -> str:
    """Single Groq call via litellm — no tools, no CrewAI loop."""
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
            if ("rate_limit_exceeded" in msg or "Rate limit" in msg) and attempt < 3:
                m = re.search(r"try again in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 5.0) if m else 20.0
                time.sleep(wait)
                continue
            raise
    return ""


def _write_report(topic: str, facts: str, groq_api_key: str) -> str:
    """Phase 2: Write the report in 4 small, adaptive, non-truncating calls."""

    system = (
        "You are a thorough research report writer. Use the provided facts. "
        "Never invent sources. Always produce every requested section fully. "
        "Never stop early."
    )

    # ---- Call 0: Decide the report structure for THIS topic -------------
    plan = _groq_call(
        groq_api_key,
        "You design report outlines. Output only the requested lines.",
        f"""
Topic: "{topic}"

Decide the BEST structure for a research report on this topic.
Then pick TWO table sections that genuinely fit the topic -- do NOT
force "Good Effects" / "Bad Effects" if they don't make sense.

Examples of good table choices:
- Pros/cons topic -> "Good Effects" and "Bad Effects"
- Historical topic -> "Timeline of Key Events" and "Key Figures / Leaders"
- Comparison topic -> "Feature Comparison" and "Pros and Cons"
- Person/org topic -> "Key Facts" and "Achievements and Controversies"
- Scientific topic -> "Benefits" and "Risks and Limitations"

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

    # ---- Call 1: Introduction + Detailed body ---------------------------
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

Output ONLY these two sections.
""",
        max_tokens=2000,
    )

    # ---- Call 2: The two adaptive tables --------------------------------
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
""",
        max_tokens=1500,
    )

    # ---- Call 3: Conclusion + Sources -----------------------------------
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
""",
        max_tokens=800,
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
            if ("rate_limit_exceeded" in msg or "Rate limit" in msg) and attempt < max_retries:
                last_error = exc
                m = re.search(r"try again in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 5.0) if m else 20.0
                time.sleep(wait)
                continue
            raise

    raise last_error  # pragma: no cover
