"""FakeOpenFoodFactsClient — in-memory OpenFoodFactsClientPort fake for unit tests."""

from snaq_verify.domain.models.source_lookup import OFFProduct
from snaq_verify.domain.ports.open_food_facts_client_port import (
    OpenFoodFactsClientPort,
)


class FakeOpenFoodFactsClient(OpenFoodFactsClientPort):
    """In-memory stub for :class:`OpenFoodFactsClientPort`.

    Pre-load barcode → product mappings and name → results mappings.
    Calls are recorded so tests can assert on invocation arguments.

    Attributes:
        barcode_calls: Every barcode string passed to ``lookup_by_barcode``.
        search_calls: Every ``(name, brand, page_size)`` tuple passed to
            ``search_by_name``.

    Example::

        product = OFFProduct(code="3017620422003", product_name="Nutella", ...)
        client = FakeOpenFoodFactsClient(
            barcode_map={"3017620422003": product},
            # Fage barcode absent → lookup returns None automatically
        )

        result = await client.lookup_by_barcode("3017620422003")
        assert result == product

        assert client.barcode_calls == ["3017620422003"]
    """

    def __init__(
        self,
        barcode_map: dict[str, OFFProduct | None] | None = None,
        search_map: dict[str, list[OFFProduct]] | None = None,
        raise_on_barcode: str | None = None,
    ) -> None:
        """Initialise the fake.

        Args:
            barcode_map: Mapping of barcode string → :class:`OFFProduct` (or
                ``None`` to simulate a 404).  Barcodes absent from the mapping
                also return ``None``.
            search_map: Mapping of product name (exact) → list of results.
                Names absent from the mapping return ``[]``.
            raise_on_barcode: If set, calling ``lookup_by_barcode`` with this
                barcode raises a :exc:`RuntimeError`.  Useful for testing
                network-error handling paths.
        """
        self._barcode_map: dict[str, OFFProduct | None] = barcode_map or {}
        self._search_map: dict[str, list[OFFProduct]] = search_map or {}
        self._raise_on_barcode = raise_on_barcode
        self.barcode_calls: list[str] = []
        self.search_calls: list[tuple[str, str | None, int]] = []

    async def lookup_by_barcode(self, barcode: str) -> OFFProduct | None:
        """Return a pre-loaded product or ``None``.

        Args:
            barcode: EAN/UPC barcode string.

        Returns:
            The configured :class:`OFFProduct`, or ``None``.

        Raises:
            RuntimeError: When ``barcode`` matches ``raise_on_barcode``.
        """
        self.barcode_calls.append(barcode)

        if self._raise_on_barcode is not None and barcode == self._raise_on_barcode:
            raise RuntimeError(
                f"FakeOpenFoodFactsClient: simulated error for barcode={barcode!r}"
            )

        return self._barcode_map.get(barcode)

    async def search_by_name(
        self,
        name: str,
        brand: str | None = None,
        page_size: int = 10,
    ) -> list[OFFProduct]:
        """Return pre-loaded search results for *name*, truncated to *page_size*.

        Args:
            name: Product name used as the lookup key.
            brand: Unused by the fake (recorded for assertion purposes only).
            page_size: Maximum number of results to return.

        Returns:
            Matching products from the pre-loaded mapping, up to *page_size*.
        """
        self.search_calls.append((name, brand, page_size))
        results = self._search_map.get(name, [])
        return results[:page_size]

    def reset(self) -> None:
        """Clear recorded calls.  Useful between test cases."""
        self.barcode_calls.clear()
        self.search_calls.clear()
