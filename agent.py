"""
Builds and runs a single-agent CrewAI "crew" that researches a topic using
free DuckDuckGo search and writes the findings up with Groq's
openai/gpt-oss-120b model.
"""

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

    # CrewAI routes non-native providers (like Groq) through LiteLLM.
    # The model string format is "groq/<model-name>".
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.5,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research the topic '{topic}' thoroughly using web search, and "
            "produce a clear, well-organized, factual report."
        ),
        backstory=(
            "You are an experienced research analyst who is excellent at "
            "turning raw web search results into clear, well-structured "
            "reports. You always search before writing, cross-check facts "
            "across multiple results, and never make up sources."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Steps to follow:\n"
            "1. Use the DuckDuckGo Search tool at least 2-3 times with "
            "different, specific search queries to gather up-to-date "
            "information on the topic.\n"
            "2. Cross-check facts that appear across multiple results.\n"
            "3. Write a clear, well-organized report in Markdown format "
            "with:\n"
            "   - A short introduction to the topic\n"
            "   - 3-5 key sections with headings covering the most "
            "important points\n"
            "   - A brief conclusion\n"
            "   - A final 'Sources' section listing the URLs you actually "
            "used\n"
        ),
        expected_output=(
            "A well-structured Markdown report, roughly 400-700 words, "
            "with headings, a conclusion, and a 'Sources' section listing "
            "real URLs returned by the search tool."
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )


def run_research(topic: str, groq_api_key: str) -> str:
    """Run the crew for a topic and return the final report as a string."""
    crew = build_crew(topic, groq_api_key)
    result = crew.kickoff()
    return str(result)
