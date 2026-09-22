"""
Builds and runs a single-agent CrewAI "crew" that researches a topic using
free DuckDuckGo search and writes the findings up with Groq's
openai/gpt-oss-120b model.

Tuned to stay well under Groq's free-tier 8,000 tokens/minute cap:
- short backstory/task text (every word costs tokens on every call)
- small search results (2 results, short snippets)
- capped completion length
- capped agent loop iterations (fewer LLM round-trips per run)
- automatic retry with backoff if the rate limit is still hit
"""

import re
import time

import litellm

from crewai import Agent, Task, Crew, Process, LLM

from tools.duckduckgo_tool import duckduckgo_search


def _patch_litellm_strip_cache_breakpoint() -> None:
    """
    Work around a current CrewAI bug: CrewAI tags messages internally with
    a `cache_breakpoint` marker to support prompt caching. That marker is
    supposed to be stripped before the request reaches non-native
    providers, but for models routed through LiteLLM (which includes Groq)
    that stripping step is currently missing, so Groq rejects the request
    with "property 'cache_breakpoint' is unsupported".

    Tracking issue: https://github.com/crewAIInc/crewAI/issues/6789
    (fix written, not yet released as of this writing)

    This patches litellm to strip the marker ourselves until CrewAI ships
    an official fix. Safe to remove once that fix is released.
    """
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


def build_crew(topic: str, groq_api_key: str) -> Crew:
    """Create the single-agent crew for a given research topic."""

    # openai/gpt-oss-120b is capped at 8,000 tokens/minute on Groq's free
    # tier. Every current free-tier Groq chat model shares roughly that
    # same cap, so the fix is to keep each run's token usage small rather
    # than pick a roomier model.
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.5,
        max_tokens=600,  # caps how long each completion can be
    )

    researcher = Agent(
        role="Research Analyst",
        goal=f"Research '{topic}' and write a short, factual report.",
        backstory="Concise research analyst. Search first, write briefly, never invent sources.",
        tools=[duckduckgo_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4,  # caps how many think/act loops the agent can take
    )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "1. Use the DuckDuckGo Search tool ONCE with a focused query "
            "(twice only if the first search genuinely isn't enough).\n"
            "2. Write a short Markdown report with:\n"
            "   - A 1-2 sentence introduction\n"
            "   - 2-3 short sections with headings\n"
            "   - A 1-2 sentence conclusion\n"
            "   - A 'Sources' section listing the URLs you used\n"
        ),
        expected_output=(
            "A concise Markdown report (250-400 words) with headings, a "
            "short conclusion, and a 'Sources' section with real URLs."
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return "rate_limit_exceeded" in msg or "Rate limit reached" in msg


def _extract_retry_seconds(exc: Exception, default: float = 20.0) -> float:
    """Groq tells us exactly how long to wait, e.g. '...in 10.4175s.'."""
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 5.0  # extra safety buffer
        except ValueError:
            pass
    return default


def run_research(topic: str, groq_api_key: str, max_retries: int = 4) -> str:
    """
    Run the crew for a topic and return the final report as a string.

    Groq's free tier limits how many tokens per minute you can use. If we
    hit that limit, wait for the time Groq tells us (plus a buffer) and
    retry automatically instead of failing the whole report. Each retry
    reruns the crew from scratch, so this is a safety net for occasional
    spikes - it isn't a substitute for keeping each run's token usage low,
    which is what the smaller prompts/results above are for.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        crew = build_crew(topic, groq_api_key)
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < max_retries:
                last_error = exc
                time.sleep(_extract_retry_seconds(exc))
                continue
            raise

    raise last_error  # pragma: no cover - unreachable in practice
