"""Tool: fetch full nutrition for a single USDA food by FDC ID."""

from agents import function_tool

from snaq_verify.domain.ports.usda_client_port import USDAClientPort


def make_get_usda_food(usda: USDAClientPort):
    """Create a USDA food-detail tool bound to *usda*.

    Returns a ``@function_tool``-decorated async callable that fetches the
    complete nutrient profile for a given FDC ID (the client is pre-bound
    via closure).

    Args:
        usda: The USDA FoodData Central client adapter.

    Returns:
        An async ``@function_tool`` that accepts ``fdc_id: int`` and returns
        a serialized ``USDACandidate`` dict with ``nutrition_per_100g``
        fully populated.

    Example::

        tool = make_get_usda_food(usda_client)
        food = await tool(171477)
    """

    @function_tool
    async def get_usda_food(fdc_id: int) -> dict:
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

    return get_usda_food
