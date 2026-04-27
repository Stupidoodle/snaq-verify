"""USDA FoodData Central HTTP client adapter."""

from __future__ import annotations

import httpx

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.models.food_item import NutritionPer100g
from snaq_verify.domain.models.source_lookup import USDACandidate
from snaq_verify.domain.ports.cache_port import CachePort
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.usda_client_port import USDAClientPort

# ---------------------------------------------------------------------------
# Nutrient ID → NutritionPer100g field mapping
# ---------------------------------------------------------------------------

#: USDA FoodData Central nutrient IDs mapped to domain model field names.
#: Reference: https://fdc.nal.usda.gov/food-details/171477/nutrients
#: Sodium FDC ID is 1093; the task prompt listed 5290 which appears to be a
#: typo — 1093 is the correct FDC nutrient number for "Sodium, Na".
NUTRIENT_ID_MAP: dict[int, str] = {
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbohydrates_g",
    1008: "calories_kcal",
    1079: "fiber_g",
    2000: "sugar_g",
    1093: "sodium_mg",
    1258: "saturated_fat_g",
}

# Accepted data-type strings from FDC (some responses use alternate spellings)
_DATA_TYPE_ALIASES: dict[str, USDADataType] = {dt.value: dt for dt in USDADataType}

# ---------------------------------------------------------------------------
# Foundation Food ID validation
# ---------------------------------------------------------------------------

#: USDA migrated Foundation Foods to the 2M+ ID range in 2024.  Older
#: Foundation IDs (pre-migration) are still indexed by /foods/search but
#: return 404 on /food/{id}.  SR Legacy and Branded IDs are unaffected.
FOUNDATION_MIN_VALID_FDC_ID: int = 2_000_000

#: The full set of nutrient IDs that must appear in a search hit for the
#: inline ``foodNutrients`` to be considered complete enough to skip a
#: follow-up ``get_food()`` call.
_REQUIRED_NUTRIENT_IDS: frozenset[int] = frozenset(NUTRIENT_ID_MAP)


def _is_likely_valid(candidate: USDACandidate) -> bool:
    """Return False for superseded Foundation Food IDs that 404 on detail.

    USDA migrated Foundation Foods to the 2M+ ID range in 2024.  Older
    Foundation IDs are still indexed by ``/foods/search`` but return 404 on
    ``/food/{id}``.  SR Legacy, Branded, and Survey IDs are unaffected.

    Args:
        candidate: The candidate to evaluate.

    Returns:
        ``True`` when the candidate is likely fetchable via ``get_food``.
        ``False`` when it is a superseded Foundation Food that will 404.
    """
    if candidate.data_type == USDADataType.FOUNDATION:
        return candidate.fdc_id >= FOUNDATION_MIN_VALID_FDC_ID
    return True


class USDAClient(USDAClientPort):
    """Concrete USDA FoodData Central adapter using ``httpx.AsyncClient``.

    Constructor arguments are injected by the bootstrap so the client never
    reads settings or instantiates dependencies itself.

    Args:
        settings: Application settings (USDA_API_KEY, USDA_BASE_URL, …).
        logger: Structured logger.
        cache: Key-value cache (checked before every HTTP call).
    """

    def __init__(
        self,
        settings: Settings,
        logger: LoggerPort,
        cache: CachePort,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._cache = cache
        self._http = httpx.AsyncClient(
            base_url=settings.USDA_BASE_URL,
            timeout=settings.HTTP_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # USDAClientPort implementation
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        data_type: USDADataType | None = None,
        page_size: int = 10,
    ) -> list[USDACandidate]:
        """Search FoodData Central by free-text query.

        Checks the cache first; on miss, calls ``/foods/search``, parses the
        response, stores the result, and returns it.

        Args:
            query: Free-text food name (e.g. "chicken breast raw").
            data_type: Optional USDA data-type filter.
            page_size: Maximum number of results to request (1–200).

        Returns:
            List of ``USDACandidate`` objects.  Empty list on zero hits.

        Raises:
            httpx.HTTPStatusError: On 4xx / 5xx responses.
            httpx.ConnectError: On network failure.
        """
        normalized = query.strip().lower()
        dt_tag = data_type.value if data_type else "all"
        cache_key = f"usda:search:{normalized}:{dt_tag}:v1"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return [USDACandidate.model_validate(c) for c in cached]

        params: dict[str, str | int] = {
            "query": query,
            "pageSize": page_size,
            "api_key": self._settings.USDA_API_KEY,
        }
        if data_type is not None:
            params["dataType"] = data_type.value

        self._logger.debug("usda search request", query=query, data_type=dt_tag)

        response = await self._http.get("/foods/search", params=params)
        self._handle_error(response, context=f"search query={query!r}")

        foods = response.json().get("foods", [])
        raw_candidates = [self._parse_search_hit(f) for f in foods]

        # Drop superseded Foundation IDs that still appear in the search index
        # but 404 on the detail endpoint.  See FOUNDATION_MIN_VALID_FDC_ID.
        candidates = [c for c in raw_candidates if _is_likely_valid(c)]
        dropped = len(raw_candidates) - len(candidates)
        if dropped:
            self._logger.debug(
                "usda search dropped superseded Foundation IDs",
                query=query,
                dropped=dropped,
            )

        self._cache.set(
            cache_key,
            [c.model_dump() for c in candidates],
            ttl_seconds=self._settings.CACHE_TTL_DAYS * 86400,
        )
        self._logger.debug(
            "usda search done", query=query, count=len(candidates)
        )
        return candidates

    async def get_food(self, fdc_id: int) -> USDACandidate:
        """Fetch the full nutrition payload for a single FDC food item.

        Args:
            fdc_id: USDA FoodData Central food identifier.

        Returns:
            ``USDACandidate`` with ``nutrition_per_100g`` populated.

        Raises:
            httpx.HTTPStatusError: On 404 or other HTTP errors.
            httpx.ConnectError: On network failure.
        """
        cache_key = f"usda:food:{fdc_id}:v1"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return USDACandidate.model_validate(cached)

        self._logger.debug("usda get_food request", fdc_id=fdc_id)

        response = await self._http.get(
            f"/food/{fdc_id}",
            params={"api_key": self._settings.USDA_API_KEY},
        )
        self._handle_error(response, context=f"get_food fdc_id={fdc_id}")

        data = response.json()
        candidate = self._parse_food_detail(data)

        self._cache.set(
            cache_key,
            candidate.model_dump(),
            ttl_seconds=self._settings.CACHE_TTL_DAYS * 86400,
        )
        self._logger.debug("usda get_food done", fdc_id=fdc_id)
        return candidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_error(self, response: httpx.Response, context: str) -> None:
        """Raise ``HTTPStatusError`` for bad responses; log 429 as warning.

        Args:
            response: The httpx response to inspect.
            context: A short description for log messages.
        """
        if response.status_code == 429:
            self._logger.warning(
                "usda rate limited",
                context=context,
                status=429,
            )
        if response.is_error:
            response.raise_for_status()

    def _parse_search_nutrients(
        self, food_nutrients: list[dict]  # type: ignore[type-arg]
    ) -> NutritionPer100g:
        """Parse the ``foodNutrients`` list from a search hit.

        The search endpoint uses ``nutrientId`` (integer) + ``value`` (float).

        Args:
            food_nutrients: The ``foodNutrients`` array from the search response.

        Returns:
            ``NutritionPer100g`` with matched fields filled; missing fields = 0.
        """
        values: dict[str, float] = {f: 0.0 for f in NUTRIENT_ID_MAP.values()}
        for entry in food_nutrients:
            nid = entry.get("nutrientId")
            if nid in NUTRIENT_ID_MAP:
                values[NUTRIENT_ID_MAP[nid]] = float(entry.get("value", 0.0))
        return NutritionPer100g(**values)

    def _parse_detail_nutrients(
        self, food_nutrients: list[dict]  # type: ignore[type-arg]
    ) -> NutritionPer100g:
        """Parse the ``foodNutrients`` list from a food-detail response.

        The detail endpoint uses ``nutrient.id`` (integer) + ``amount`` (float).

        Args:
            food_nutrients: The ``foodNutrients`` array from the detail response.

        Returns:
            ``NutritionPer100g`` with matched fields filled; missing fields = 0.
        """
        values: dict[str, float] = {f: 0.0 for f in NUTRIENT_ID_MAP.values()}
        for entry in food_nutrients:
            nutrient = entry.get("nutrient", {})
            nid = nutrient.get("id")
            if nid in NUTRIENT_ID_MAP:
                values[NUTRIENT_ID_MAP[nid]] = float(entry.get("amount", 0.0))
        return NutritionPer100g(**values)

    def _parse_search_hit(
        self, food: dict  # type: ignore[type-arg]
    ) -> USDACandidate:
        """Convert one FDC search-result dict into a ``USDACandidate``.

        Populates ``nutrition_per_100g`` from the inline ``foodNutrients``
        array when **all** eight mapped nutrient IDs are present, avoiding a
        follow-up ``get_food()`` call for happy-path queries.  When the inline
        data is incomplete (USDA omits some nutrients from search hits),
        ``nutrition_per_100g`` is left ``None`` so the agent knows to call
        ``get_food()`` for authoritative values.

        Args:
            food: A single entry from the ``foods`` array in the search response.

        Returns:
            ``USDACandidate`` with ``nutrition_per_100g`` set when the inline
            nutrient list is complete, otherwise ``None``.
        """
        raw_dt = food.get("dataType", "Foundation")
        data_type = _DATA_TYPE_ALIASES.get(raw_dt, USDADataType.FOUNDATION)

        food_nutrients: list[dict] = food.get("foodNutrients", [])  # type: ignore[type-arg]
        present_ids = {int(e["nutrientId"]) for e in food_nutrients if "nutrientId" in e}
        nutrition: NutritionPer100g | None = (
            self._parse_search_nutrients(food_nutrients)
            if _REQUIRED_NUTRIENT_IDS.issubset(present_ids)
            else None
        )

        return USDACandidate(
            fdc_id=food["fdcId"],
            description=food.get("description", ""),
            data_type=data_type,
            brand_owner=food.get("brandOwner"),
            nutrition_per_100g=nutrition,
            relevance_score=food.get("score"),
        )

    def _parse_food_detail(
        self, data: dict  # type: ignore[type-arg]
    ) -> USDACandidate:
        """Convert a FDC food-detail dict into a ``USDACandidate``.

        Args:
            data: The full JSON response from ``GET /food/{fdcId}``.

        Returns:
            ``USDACandidate`` with ``nutrition_per_100g`` populated.
        """
        raw_dt = data.get("dataType", "Foundation")
        data_type = _DATA_TYPE_ALIASES.get(raw_dt, USDADataType.FOUNDATION)
        nutrition = self._parse_detail_nutrients(data.get("foodNutrients", []))
        return USDACandidate(
            fdc_id=data["fdcId"],
            description=data.get("description", ""),
            data_type=data_type,
            brand_owner=data.get("brandOwner"),
            nutrition_per_100g=nutrition,
        )
