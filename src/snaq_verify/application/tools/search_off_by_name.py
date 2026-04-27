"""Tool: search Open Food Facts by product name (and optional brand)."""

from collections.abc import Awaitable, Callable

from agents import function_tool

from snaq_verify.domain.models.source_lookup import OFFProduct
from snaq_verify.domain.ports.open_food_facts_client_port import (
    OpenFoodFactsClientPort,
)


def make_search_off_by_name(
    off: OpenFoodFactsClientPort,
) -> tuple[Callable[..., Awaitable[list[OFFProduct]]], object]:
    """Create a name-search callable bound to *off*.

    Returns a 2-tuple of ``(raw_fn, function_tool_wrapper)``:

    * ``raw_fn`` — a plain async callable; tests call this directly without
      going through the agents tool runner.
    * ``function_tool_wrapper`` — a :class:`~agents.FunctionTool` ready for
      use in ``Agent(tools=[...])``.  The agent factory (verifier agent adapter)
      unpacks and uses this.

    Args:
        off: The Open Food Facts client adapter.

    Returns:
        ``(search_off_by_name, search_off_by_name_tool)``

    Example::

        fn, tool = make_search_off_by_name(off_client)

        # In tests — call the raw function directly:
        products = await fn("Total 0% Greek Yogurt", brand="Fage")

        # In the agent adapter — register the tool:
        agent = Agent(tools=[tool, ...])
    """

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

    # Pre-built FunctionTool for Agent(tools=[...]).
    # Tests call `search_off_by_name(...)` directly; agent-domain imports the tool.
    search_off_by_name_tool = function_tool(search_off_by_name)

    return search_off_by_name, search_off_by_name_tool
