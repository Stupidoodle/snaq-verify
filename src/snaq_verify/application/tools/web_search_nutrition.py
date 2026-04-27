"""Web search nutrition tool — Tavily-backed last-resort fallback."""

from agents import function_tool

from snaq_verify.domain.models.source_lookup import WebSnippet
from snaq_verify.domain.ports.tavily_client_port import TavilyClientPort

# Module-level client reference, injected at bootstrap time via
# ``configure_web_search``.  Using a module-level singleton avoids threading
# the client through every function signature while keeping the tool
# stateless from the agent's perspective.
_tavily_client: TavilyClientPort | None = None


def configure_web_search(client: TavilyClientPort | None) -> None:
    """Inject the Tavily client used by :func:`web_search_nutrition`.

    Should be called once during application bootstrap (see ``bootstrap.py``)
    and during test setup to inject a :class:`FakeTavilyClient`.

    Args:
        client: A :class:`TavilyClientPort` implementation, or ``None`` to
            reset (useful in test teardown).
    """
    global _tavily_client
    _tavily_client = client


@function_tool
async def web_search_nutrition(query: str, max_results: int = 5) -> list[WebSnippet]:
    """Search the web for nutrition information using Tavily.

    This tool is the **last-resort fallback** in the verification pipeline —
    used only when neither USDA nor Open Food Facts can supply sufficient data.
    It returns ranked content snippets that the agent can scan to extract or
    validate nutrient figures.

    Configure the underlying client at startup by calling
    :func:`configure_web_search` before invoking this tool.

    Args:
        query: Free-text search query.  Examples:
            ``"Fage Total 0% Greek Yogurt nutrition per 100g"``
            ``"avocado raw macros protein fat carbohydrates"``
        max_results: Maximum number of snippets to return (default 5).

    Returns:
        Ranked :class:`WebSnippet` objects — may be empty when no results
        match the query.

    Raises:
        RuntimeError: If :func:`configure_web_search` has not been called.
        Exception: Any transport error from the Tavily client propagates
            unchanged.
    """
    if _tavily_client is None:
        raise RuntimeError(
            "web_search_nutrition tool is not configured — "
            "call configure_web_search(client) during application bootstrap."
        )
    return await _tavily_client.search(query, max_results=max_results)
