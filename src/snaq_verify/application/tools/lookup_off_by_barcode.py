"""Tool: look up a product on Open Food Facts by barcode."""

from agents import function_tool

from snaq_verify.domain.models.source_lookup import OFFProduct
from snaq_verify.domain.ports.open_food_facts_client_port import (
    OpenFoodFactsClientPort,
)


def make_lookup_off_by_barcode(off: OpenFoodFactsClientPort):
    """Create a barcode-lookup tool bound to *off*.

    Returns a ``@function_tool``-decorated async callable that the verifier
    agent can invoke with only the ``barcode`` argument (the client is
    pre-bound via closure).

    Args:
        off: The Open Food Facts client adapter.

    Returns:
        An async ``@function_tool`` that accepts ``barcode: str`` and returns
        the matching :class:`OFFProduct` or ``None``.

    Example::

        tool = make_lookup_off_by_barcode(off_client)
        product = await tool("3017620422003")
    """

    @function_tool
    async def lookup_off_by_barcode(barcode: str) -> OFFProduct | None:
        """Look up a product on Open Food Facts by exact EAN/UPC barcode.

        Prefer this over ``search_off_by_name`` when a barcode is available —
        it is deterministic and faster.  Returns ``None`` when the barcode is
        not in the OFF database (e.g., the Fage ``5200435000027`` case); in
        that situation fall back to ``search_off_by_name``.

        Args:
            barcode: The EAN-13 or UPC-A barcode string (digits only).

        Returns:
            The :class:`OFFProduct` with nutrition data when found, or
            ``None`` on a 404 miss.
        """
        return await off.lookup_by_barcode(barcode)

    return lookup_off_by_barcode
