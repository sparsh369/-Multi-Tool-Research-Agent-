"""Tools the research agent can call: web search and page reading.

Each tool is decorated with @tool so LangGraph/LangChain can expose its
signature and docstring to the LLM for function-calling.
"""
from __future__ import annotations

import requests
from langchain_core.tools import tool

from .config import SEARCH_RESULTS_PER_QUERY, MAX_PAGE_CHARS

# DuckDuckGo search lib was renamed from `duckduckgo_search` to `ddgs`.
# Support both so the project works across versions.
try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore


@tool
def web_search(query: str) -> str:
    """Search the web for a query and return a list of result titles, URLs and snippets.

    Use this to discover sources. Follow up with `read_url` to read the most
    promising results in full. Returns up to a handful of results.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=SEARCH_RESULTS_PER_QUERY))
    except Exception as exc:  # network / rate-limit resilience
        return f"SEARCH_ERROR: {exc}"

    if not results:
        return "No results found. Try rephrasing the query."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        url = r.get("href") or r.get("url", "")
        snippet = r.get("body", "")
        lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")
    return "\n".join(lines)


@tool
def read_url(url: str) -> str:
    """Fetch a web page and return its main text content, cleaned of navigation/ads.

    Use this after `web_search` to read a promising source in depth before
    citing it. Content is truncated to keep token cost bounded.
    """
    import trafilatura

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
        )
        resp.raise_for_status()
    except Exception as exc:
        return f"FETCH_ERROR for {url}: {exc}"

    text = trafilatura.extract(resp.text) or ""
    if not text.strip():
        return f"NO_READABLE_CONTENT at {url} (page may be JS-rendered or blocked)."

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n...[truncated]"
    return f"CONTENT FROM {url}:\n{text}"


TOOLS = [web_search, read_url]
