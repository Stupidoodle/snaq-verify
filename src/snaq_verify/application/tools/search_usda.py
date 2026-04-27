"""Factory for the search_usda agent tool."""

from __future__ import annotations

from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.ports.usda_client_port import USDAClientPort


def make_search_usda(usda: USDAClientPort):
    """Return a raw async callable bound to *usda* for searching USDA FoodData Central.

    The returned coroutine function can be tested directly with
    ``await tool("query")`` or wrapped with
    ``function_tool(make_search_usda(client))`` for use in an
    OpenAI Agents SDK ``Agent``.

    Args:
        usda: The USDA FoodData Central client adapter.

    Returns:
        An async function ``search_usda(query, data_type, page_size) -> list[dict]``.

    Example::

        tool = make_search_usda(usda_client)
        results = await tool("chicken breast", data_type="Foundation")
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

    return search_usda
