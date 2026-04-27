"""FakeUSDAClient — in-memory USDAClientPort stub for unit tests."""

from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.models.source_lookup import USDACandidate
from snaq_verify.domain.ports.usda_client_port import USDAClientPort


class FakeUSDAClient(USDAClientPort):
    """In-memory stub for :class:`USDAClientPort`.

    Pre-load canned search results and food records via :meth:`add_search_result`
    and :meth:`add_food`. Every call is recorded so tests can assert on
    arguments passed.

    Attributes:
        search_calls: List of dicts with keys ``query``, ``data_type``,
            ``page_size`` — one entry per :meth:`search` call.
        get_food_calls: List of ``fdc_id`` integers — one per :meth:`get_food`
            call.

    Example::

        client = FakeUSDAClient()
        client.add_search_result("chicken", [some_candidate])
        client.add_food(171477, detailed_candidate)

        results = await client.search("chicken")
        assert results == [some_candidate]
        food = await client.get_food(171477)
    """

    def __init__(
        self,
        raise_on: str | None = None,
        raise_on_fdc_id: int | None = None,
    ) -> None:
        """Initialise the fake.

        Args:
            raise_on: If set, calling :meth:`search` with this exact query will
                raise :class:`RuntimeError` — useful for testing error paths.
            raise_on_fdc_id: If set, calling :meth:`get_food` with this FDC ID
                will raise :class:`RuntimeError`.
        """
        self._search_results: dict[str, list[USDACandidate]] = {}
        self._food_results: dict[int, USDACandidate] = {}
        self._raise_on = raise_on
        self._raise_on_fdc_id = raise_on_fdc_id
        self.search_calls: list[dict] = []
        self.get_food_calls: list[int] = []

    # ------------------------------------------------------------------
    # Setup helpers (call these before the system-under-test)
    # ------------------------------------------------------------------

    def add_search_result(self, query: str, results: list[USDACandidate]) -> None:
        """Pre-load *results* for the given *query* string.

        Args:
            query: The exact query string the client should match on.
            results: The :class:`USDACandidate` list to return.
        """
        self._search_results[query] = results

    def add_food(self, fdc_id: int, candidate: USDACandidate) -> None:
        """Pre-load a *candidate* for :meth:`get_food` to return.

        Args:
            fdc_id: The FDC ID key.
            candidate: The :class:`USDACandidate` to return.
        """
        self._food_results[fdc_id] = candidate

    # ------------------------------------------------------------------
    # USDAClientPort implementation
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        data_type: USDADataType | None = None,
        page_size: int = 10,
    ) -> list[USDACandidate]:
        """Return pre-loaded results for *query*, or an empty list.

        Args:
            query: Free-text food name.
            data_type: Optional data-type filter (recorded but not filtered).
            page_size: Maximum results (recorded; pre-loaded list is truncated
                to this value).

        Returns:
            Matching pre-loaded candidates, up to *page_size*.

        Raises:
            RuntimeError: When ``query`` matches the ``raise_on`` argument
                given at construction time.
        """
        self.search_calls.append(
            {"query": query, "data_type": data_type, "page_size": page_size}
        )

        if self._raise_on is not None and query == self._raise_on:
            raise RuntimeError(
                f"FakeUSDAClient: simulated error for query={query!r}"
            )

        results = self._search_results.get(query, [])
        return results[:page_size]

    async def get_food(self, fdc_id: int) -> USDACandidate:
        """Return the pre-loaded candidate for *fdc_id*.

        Args:
            fdc_id: USDA food identifier.

        Returns:
            The pre-loaded :class:`USDACandidate`.

        Raises:
            RuntimeError: When *fdc_id* matches the ``raise_on_fdc_id``
                argument given at construction.
            KeyError: When *fdc_id* has no pre-loaded result — signals a
                programming error in the test setup.
        """
        self.get_food_calls.append(fdc_id)

        if self._raise_on_fdc_id is not None and fdc_id == self._raise_on_fdc_id:
            raise RuntimeError(
                f"FakeUSDAClient: simulated error for fdc_id={fdc_id}"
            )

        if fdc_id not in self._food_results:
            raise KeyError(f"FakeUSDAClient: no pre-loaded food for fdc_id={fdc_id}")

        return self._food_results[fdc_id]

    def reset(self) -> None:
        """Clear all recorded calls.  Useful between test cases."""
        self.search_calls.clear()
        self.get_food_calls.clear()
