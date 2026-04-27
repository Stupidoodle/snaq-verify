"""Tool: search USDA FoodData Central by free-text query."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agents import function_tool

from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.ports.usda_client_port import USDAClientPort


def make_search_usda(
    usda: USDAClientPort,
) -> tuple[Callable[..., Awaitable[list[dict]]], object]:  # type: ignore[type-arg]
    """Create a USDA search tool bound to *usda*.

    Returns a 2-tuple of ``(raw_fn, function_tool_wrapper)``:

    * ``raw_fn`` — a plain async callable; tests call this directly without
      going through the agents tool runner.
    * ``function_tool_wrapper`` — a :class:`~agents.FunctionTool` ready for
      use in ``Agent(tools=[...])``.

    Args:
        usda: The USDA FoodData Central client adapter.

    Returns:
        ``(search_usda, search_usda_tool)``

    Example::

        fn, tool = make_search_usda(usda_client)

        # In tests — call the raw function directly:
        results = await fn("chicken breast", data_type="Foundation")

        # In the agent adapter — register the tool:
        agent = Agent(tools=[tool, ...])
    """

    async def search_usda(
        query: str,
        data_type: str | None = None,
        page_size: int = 10,
    ) -> list[dict]:  # type: ignore[type-arg]
        """Search USDA FoodData Central for foods matching the query string.

        Use this tool to find candidate food records in the USDA database.
        For generic raw foods, prefer filtering by ``data_type="Foundation"``
        or ``"SR Legacy"`` to get the most reliable nutrient values.  Call
        ``get_usda_food`` afterwards to retrieve the full nutrient profile for
        a specific FDC ID found here.

        Args:
            query: Free-text food name, e.g. ``"chicken breast raw"``.
            data_type: Optional USDA data-type filter — one of
                ``"Foundation"``, ``"SR Legacy"``, ``"Branded"``,
                ``"Survey (FNDDS)"``.  Omit to search all types.
            page_size: Maximum number of results (1–200). Default 10.

        Returns:
            List of candidate dicts, each with keys ``fdc_id``,
            ``description``, ``data_type``, ``brand_owner``,
            ``nutrition_per_100g``, and ``relevance_score``.  Empty list
            when no matches.
        """
        dt = USDADataType(data_type) if data_type is not None else None
        candidates = await usda.search(query, data_type=dt, page_size=page_size)
        return [c.model_dump() for c in candidates]

    search_usda_tool = function_tool(search_usda)
    return search_usda, search_usda_tool
