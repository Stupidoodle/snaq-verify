"""Open Food Facts client port."""

from abc import ABC, abstractmethod

from snaq_verify.domain.models.source_lookup import OFFProduct


class OpenFoodFactsClientPort(ABC):
    """Abstract interface for the Open Food Facts client.

    Two operations: `lookup_by_barcode` (preferred when a barcode is known —
    deterministic 1:1 lookup) and `search_by_name` (fallback for products
    that aren't in OFF or whose barcode varies by region).
    """

    @abstractmethod
    async def lookup_by_barcode(self, barcode: str) -> OFFProduct | None:
        """Fetch a product by exact barcode.

        Args:
            barcode: EAN/UPC barcode string.

        Returns:
            The product, or None when OFF returns 404 (the Fage `5200435000027`
            case from the input file is one of these — fall back to
            `search_by_name`).
        """
        raise NotImplementedError

    @abstractmethod
    async def search_by_name(
        self,
        name: str,
        brand: str | None = None,
        page_size: int = 10,
    ) -> list[OFFProduct]:
        """Search OFF by product name (and optionally brand).

        Args:
            name: Product name (e.g., "Total 0% Greek Yogurt").
            brand: Optional brand filter (e.g., "Fage").
            page_size: Max results to return.

        Returns:
            Ranked products. Empty list when no matches.
        """
        raise NotImplementedError
