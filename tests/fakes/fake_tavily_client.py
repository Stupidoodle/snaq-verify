"""FakeTavilyClient — in-memory TavilyClientPort fake for unit tests."""

from snaq_verify.domain.models.source_lookup import WebSnippet
from snaq_verify.domain.ports.tavily_client_port import TavilyClientPort


class FakeTavilyClient(TavilyClientPort):
    """In-memory stub for :class:`TavilyClientPort`.

    Pre-load canned responses by query string.  The ``search`` method returns
    the matching list (truncated to *max_results*) or an empty list when no
    match is found.

    Attributes:
        calls: Records of every ``search`` invocation as ``(query, max_results)``
            tuples.  Useful for asserting call counts and arguments.

    Example::

        snippet = WebSnippet(url=\"https://example.com\", title=\"Chicken\",
                             content=\"Protein 31g per 100g\", score=0.95)
        client = FakeTavilyClient(responses={\"chicken nutrition\": [snippet]})

        results = await client.search(\"chicken nutrition\", max_results=3)
        assert results == [snippet]

        assert client.calls == [(\"chicken nutrition\", 3)]
    """

    def __init__(
        self,
        responses: dict[str, list[WebSnippet]] | None = None,
        raise_on: str | None = None,
    ) -> None:
        """Initialise the fake.

        Args:
            responses: Mapping of query string → list of ``WebSnippet``.
                Queries not in the mapping return ``[]``.
            raise_on: If set, calling ``search`` with this exact query string
                will raise a ``RuntimeError``.  Useful for testing error paths.
        """
        self._responses: dict[str, list[WebSnippet]] = responses or {}
        self._raise_on = raise_on
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int = 5) -> list[WebSnippet]:
        """Return canned results for *query*, truncated to *max_results*.

        Args:
            query: The search query.
            max_results: Maximum number of snippets to return.

        Returns:
            Matching snippets from the pre-loaded responses, up to *max_results*.

        Raises:
            RuntimeError: When ``query`` matches ``raise_on``.
        """
        self.calls.append((query, max_results))

        if self._raise_on is not None and query == self._raise_on:
            raise RuntimeError(f"FakeTavilyClient: simulated error for query={query!r}")

        results = self._responses.get(query, [])
        return results[:max_results]

    def reset(self) -> None:
        """Clear recorded calls.  Useful between test cases."""
        self.calls.clear()
