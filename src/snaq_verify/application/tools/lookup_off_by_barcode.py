"""Tool: look up a product on Open Food Facts by barcode."""

from collections.abc import Awaitable, Callable

from agents import function_tool

from snaq_verify.domain.models.source_lookup import OFFProduct
from snaq_verify.domain.ports.open_food_facts_client_port import (
    OpenFoodFactsClientPort,
)


def make_lookup_off_by_barcode(
    off: OpenFoodFactsClientPort,
) -> tuple[Callable[[str], Awaitable[OFFProduct | None]], object]:
    """Create a barcode-lookup callable bound to *off*.

    Returns a 2-tuple of ``(raw_fn, function_tool_wrapper)``:

    * ``raw_fn`` — a plain async callable; tests call this directly without
      going through the agents tool runner.
    * ``function_tool_wrapper`` — a :class:`~agents.FunctionTool` ready for
      use in ``Agent(tools=[...])``.  The agent factory (verifier agent adapter)
      unpacks and uses this.

    Args:
        off: The Open Food Facts client adapter.

    Returns:
        ``(lookup_off_by_barcode, lookup_off_by_barcode_tool)``

    Example::

        fn, tool = make_lookup_off_by_barcode(off_client)

        # In tests — call the raw function directly:
        product = await fn("3017620422003")

        # In the agent adapter — register the tool:
        agent = Agent(tools=[tool, ...])
    """

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

    # Pre-built FunctionTool for Agent(tools=[...]).
    # Tests call `lookup_off_by_barcode(...)` directly; agent-domain imports the tool.
    lookup_off_by_barcode_tool = function_tool(lookup_off_by_barcode)

    return lookup_off_by_barcode, lookup_off_by_barcode_tool
