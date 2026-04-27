"""Tavily web search client port."""

from abc import ABC, abstractmethod

from snaq_verify.domain.models.source_lookup import WebSnippet


class TavilyClientPort(ABC):
    """Abstract interface for the Tavily search client.

    Used as the fallback when USDA + OFF both miss. Returns content snippets
    inline (Tavily's default) so the agent can scan them without a separate
    fetch step.
    """

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[WebSnippet]:
        """Run a web search and return ranked snippets.

        Args:
            query: Free-text query (e.g.,
                "Fage Total 0% Greek Yogurt nutrition per 100g").
            max_results: Maximum snippets to return.

        Returns:
            Ranked snippets, each with title/url/content/score. Empty list on
            no matches.
        """
        raise NotImplementedError
