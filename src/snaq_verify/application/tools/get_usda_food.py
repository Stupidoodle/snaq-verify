"""Tool: fetch full nutrition for a single USDA food by FDC ID."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agents import function_tool

from snaq_verify.domain.ports.usda_client_port import USDAClientPort


def make_get_usda_food(
    usda: USDAClientPort,
) -> tuple[Callable[..., Awaitable[dict]], object]:  # type: ignore[type-arg]
    """Create a USDA food-detail tool bound to *usda*.

    Returns a 2-tuple of ``(raw_fn, function_tool_wrapper)``:

    * ``raw_fn`` — a plain async callable; tests call this directly.
    * ``function_tool_wrapper`` — a :class:`~agents.FunctionTool` ready for
      use in ``Agent(tools=[...])``.

    Args:
        usda: The USDA FoodData Central client adapter.

    Returns:
        ``(get_usda_food, get_usda_food_tool)``

    Example::

        fn, tool = make_get_usda_food(usda_client)

        # In tests:
        food = await fn(171477)

        # In the agent adapter:
        agent = Agent(tools=[tool, ...])
    """

    async def get_usda_food(fdc_id: int) -> dict:  # type: ignore[type-arg]
        """Fetch the complete nutrient profile for a USDA FDC food item.

        Use this after ``search_usda`` to retrieve the authoritative
        per-100g nutrition for a specific ``fdcId``.  The detail endpoint
        returns a full nutrient list, whereas the search endpoint may return
        an incomplete one.

        Args:
            fdc_id: USDA FoodData Central food identifier (integer), obtained
                from a prior ``search_usda`` result.

        Returns:
            A dict with keys ``fdc_id``, ``description``, ``data_type``,
            ``brand_owner``, ``nutrition_per_100g``, and ``relevance_score``.
            The ``nutrition_per_100g`` sub-dict contains all eight nutrient
            fields, defaulting absent nutrients to ``0.0``.

        Raises:
            httpx.HTTPStatusError: When FDC returns 404 (food not found) or
                another HTTP error.
        """
        candidate = await usda.get_food(fdc_id)
        return candidate.model_dump()

    get_usda_food_tool = function_tool(get_usda_food)
    return get_usda_food, get_usda_food_tool
