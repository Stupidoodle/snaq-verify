"""Tool: search Open Food Facts by product name (and optional brand)."""

from agents import function_tool

from snaq_verify.domain.models.source_lookup import OFFProduct
from snaq_verify.domain.ports.open_food_facts_client_port import (
    OpenFoodFactsClientPort,
)


def make_search_off_by_name(off: OpenFoodFactsClientPort):
    """Create a name-search tool bound to *off*.

    Returns a ``@function_tool``-decorated async callable that the verifier
    agent can invoke with ``name``, optional ``brand``, and optional
    ``page_size`` (the client is pre-bound via closure).

    Args:
        off: The Open Food Facts client adapter.

    Returns:
        An async ``@function_tool`` that accepts ``name: str``,
        ``brand: str | None``, and ``page_size: int`` and returns a
        list of :class:`OFFProduct` objects.

    Example::

        tool = make_search_off_by_name(off_client)
        products = await tool("Total 0% Greek Yogurt", brand="Fage")
    """

    @function_tool
    async def search_off_by_name(
        name: str,
        brand: str | None = None,
        page_size: int = 10,
    ) -> list[OFFProduct]:
        """Search Open Food Facts by product name, with an optional brand filter.

        Use this as a fallback when ``lookup_off_by_barcode`` returns ``None``,
        or when no barcode is available.

        Args:
            name: Product name to search for (e.g., ``"Total 0% Greek Yogurt"``).
            brand: Optional brand name to narrow results (e.g., ``"Fage"``).
            page_size: Maximum number of results to return (default 10).

        Returns:
            A list of :class:`OFFProduct` objects ranked by OFF's relevance
            score.  May be empty when no products match.
        """
        return await off.search_by_name(name, brand=brand, page_size=page_size)

    return search_off_by_name
