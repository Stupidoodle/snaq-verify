"""Tavily web-search client adapter."""

from tavily import AsyncTavilyClient

from snaq_verify.domain.models.source_lookup import WebSnippet
from snaq_verify.domain.ports.cache_port import CachePort
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.tavily_client_port import TavilyClientPort

#: Default cache TTL: 30 days in seconds.
DEFAULT_TTL_SECONDS: int = 30 * 86_400


class TavilyClient(TavilyClientPort):
    """Tavily web-search adapter backed by the official ``tavily-python`` SDK.

    Results are cached keyed by ``tavily:{query}:{max_results}:v1`` so
    repeated identical queries are free.  Namespacing is the adapter's
    responsibility — callers don't need to add a prefix.

    Args:
        api_key: Tavily API key (from ``Settings.TAVILY_API_KEY``).
        cache: Cache adapter (``FileCache`` in production, ``InMemoryCache``
            in tests).
        logger: Structured logger.
        ttl_seconds: Cache TTL in seconds.  Defaults to 30 days.
    """

    def __init__(
        self,
        api_key: str,
        cache: CachePort,
        logger: LoggerPort,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._client = AsyncTavilyClient(api_key=api_key)
        self._cache = cache
        self._logger = logger
        self._ttl = ttl_seconds

    async def search(self, query: str, max_results: int = 5) -> list[WebSnippet]:
        """Run a web search and return ranked snippets.

        Checks the cache first.  On a miss, calls Tavily, stores the result,
        then returns it.

        Args:
            query: Free-text query (e.g.,
                ``"Fage Total 0% Greek Yogurt nutrition per 100g"``).
            max_results: Maximum snippets to return (forwarded to the API).

        Returns:
            Ranked ``WebSnippet`` objects — may be empty.

        Raises:
            Exception: Any transport error from the Tavily SDK propagates
                unchanged.  Callers should decide retry/fallback behaviour.
        """
        normalized_query = query.strip().lower()
        cache_key = f"tavily:{normalized_query}:{max_results}:v1"

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._logger.debug(
                "tavily cache hit",
                query=query,
                max_results=max_results,
                cache_key=cache_key,
            )
            return [WebSnippet(**item) for item in cached]

        self._logger.debug(
            "tavily cache miss — calling API",
            query=query,
            max_results=max_results,
        )

        try:
            response: dict = await self._client.search(query, max_results=max_results)
        except Exception:
            self._logger.error(
                "tavily search failed",
                query=query,
                max_results=max_results,
            )
            raise

        raw_results: list[dict] = response.get("results", [])
        snippets = [
            WebSnippet(
                url=r.get("url", ""),
                title=r.get("title", ""),
                content=r.get("content", ""),
                score=r.get("score"),
            )
            for r in raw_results
        ]

        self._cache.set(
            cache_key,
            [s.model_dump() for s in snippets],
            ttl_seconds=self._ttl,
        )

        self._logger.info(
            "tavily search complete",
            query=query,
            max_results=max_results,
            result_count=len(snippets),
        )

        return snippets
