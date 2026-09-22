"""
A free web search tool for our CrewAI agent, built on top of the `ddgs`
package (the renamed/maintained successor of `duckduckgo-search`).

No API key is required for this tool — that's what makes it beginner
friendly and free to run.
"""

from crewai.tools import tool
from ddgs import DDGS


@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return the top results.

    Use this whenever you need current, factual information about a topic —
    for example, recent news, statistics, definitions, or general
    background. Call it several times with different, specific queries to
    build up a well-rounded picture of the topic before writing anything.

    Args:
        query: A short, specific search query (a few words works best,
            e.g. "solid-state battery 2026 breakthroughs").

    Returns:
        A numbered list of results, each with a title, URL, and short
        snippet, or a message saying no results were found.
    """
    # Kept small on purpose: Groq's free tier has a strict tokens-per-minute
    # limit, and every character returned here becomes tokens the LLM has
    # to read. 3 results with short snippets is usually enough context.
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as exc:  # network hiccups, rate limits, etc.
        return f"Search failed for query '{query}': {exc}"

    if not results:
        return f"No results found for query: '{query}'"

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        link = r.get("href", "No link")
        snippet = (r.get("body", "") or "")[:220]  # truncate long snippets
        formatted.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")

    return "\n\n".join(formatted)
