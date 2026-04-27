"""Open Food Facts HTTP client adapter."""

from typing import Any

import httpx
from pydantic import ValidationError

from snaq_verify.core.config import Settings
from snaq_verify.domain.models.food_item import NutritionPer100g
from snaq_verify.domain.models.source_lookup import OFFProduct
from snaq_verify.domain.ports.cache_port import CachePort
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.open_food_facts_client_port import (
    OpenFoodFactsClientPort,
)

#: Sentinel stored in cache when a barcode lookup returns 404, so we don't
#: re-hit the network on the next call for the same (absent) barcode.
_NOT_FOUND_SENTINEL = "__not_found__"


def _parse_nutriments(nutriments: dict[str, Any]) -> NutritionPer100g | None:
    """Convert a raw OFF ``nutriments`` dict to :class:`NutritionPer100g`.

    Args:
        nutriments: The ``nutriments`` sub-dict from an OFF product JSON.

    Returns:
        A fully populated :class:`NutritionPer100g`, or ``None`` when any
        required field is absent or when a Pydantic validation error occurs
        (e.g., ``calories_kcal > 900`` hinting at a kJ/kcal unit confusion).
    """
    simple_keys: dict[str, str] = {
        "energy-kcal_100g": "calories_kcal",
        "proteins_100g": "protein_g",
        "fat_100g": "fat_g",
        "saturated-fat_100g": "saturated_fat_g",
        "carbohydrates_100g": "carbohydrates_g",
        "sugars_100g": "sugar_g",
        "fiber_100g": "fiber_g",
    }

    values: dict[str, float] = {}
    for off_key, model_field in simple_keys.items():
        raw = nutriments.get(off_key)
        if raw is None:
            return None
        try:
            values[model_field] = float(raw)
        except (TypeError, ValueError):
            return None

    # Sodium: OFF stores in grams per 100 g; the model expects milligrams.
    # Prefer ``sodium_100g``; fall back to ``salt_100g`` (salt = sodium * 2.5).
    sodium_raw = nutriments.get("sodium_100g")
    if sodium_raw is not None:
        try:
            values["sodium_mg"] = float(sodium_raw) * 1000.0
        except (TypeError, ValueError):
            return None
    else:
        salt_raw = nutriments.get("salt_100g")
        if salt_raw is None:
            return None
        try:
            # salt_g / 2.5 = sodium_g; * 1000 = sodium_mg
            values["sodium_mg"] = float(salt_raw) * 1000.0 / 2.5
        except (TypeError, ValueError):
            return None

    try:
        return NutritionPer100g(**values)
    except ValidationError:
        return None


def _product_from_raw(raw: dict[str, Any]) -> OFFProduct:
    """Build an :class:`OFFProduct` from an OFF product JSON dict.

    Args:
        raw: A single product dict as returned by the OFF API.

    Returns:
        An :class:`OFFProduct` with ``nutrition_per_100g`` populated when
        OFF supplies complete nutriment data.
    """
    nutriments = raw.get("nutriments") or {}
    nutrition = _parse_nutriments(nutriments) if nutriments else None

    popularity_key_raw = raw.get("popularity_key")
    try:
        popularity_key: int | None = int(popularity_key_raw) if popularity_key_raw is not None else None
    except (TypeError, ValueError):
        popularity_key = None

    completeness_raw = raw.get("completeness")
    try:
        completeness: float | None = float(completeness_raw) if completeness_raw is not None else None
    except (TypeError, ValueError):
        completeness = None

    return OFFProduct(
        code=str(raw.get("code", raw.get("_id", ""))),
        product_name=raw.get("product_name") or None,
        brands=raw.get("brands") or None,
        nutrition_per_100g=nutrition,
        completeness=completeness,
        popularity_key=popularity_key,
    )


class OpenFoodFactsClient(OpenFoodFactsClientPort):
    """Async HTTP adapter for the Open Food Facts public API.

    Uses :mod:`httpx` for async HTTP (the official ``openfoodfacts`` Python
    package uses the sync ``requests`` library, which is not compatible with
    the project's async pipeline).

    Results are cached to avoid redundant network calls:

    * Barcode lookup: ``off:barcode:{barcode}:v1``
    * Name search:    ``off:search:{name_lower}:{brand_lower}:{page_size}:v1``

    404 responses (e.g., the Fage ``5200435000027`` barcode) are cached as the
    sentinel ``"__not_found__"`` so repeated lookups don't re-hit the network.

    Args:
        settings: Application settings (provides base URL, user-agent, TTL).
        logger: Structured logger.
        cache: Cache adapter.
    """

    def __init__(
        self,
        settings: Settings,
        logger: LoggerPort,
        cache: CachePort,
    ) -> None:
        self._base_url = settings.OFF_BASE_URL.rstrip("/")
        self._user_agent = settings.OFF_USER_AGENT
        self._timeout = settings.HTTP_TIMEOUT_SECONDS
        self._ttl = settings.CACHE_TTL_DAYS * 86_400
        self._logger = logger
        self._cache = cache

    # ------------------------------------------------------------------
    # OpenFoodFactsClientPort interface
    # ------------------------------------------------------------------

    async def lookup_by_barcode(self, barcode: str) -> OFFProduct | None:
        """Fetch a product by exact barcode.

        Returns ``None`` when OFF returns 404 (e.g., Fage ``5200435000027``).

        Args:
            barcode: EAN/UPC barcode string.

        Returns:
            The :class:`OFFProduct`, or ``None`` on a 404 miss.

        Raises:
            httpx.HTTPStatusError: On non-404 HTTP errors.
            httpx.TransportError: On network-level failures.
        """
        cache_key = f"off:barcode:{barcode}:v1"

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._logger.debug(
                "off barcode cache hit",
                barcode=barcode,
                cache_key=cache_key,
            )
            if cached == _NOT_FOUND_SENTINEL:
                return None
            return OFFProduct(**cached)

        self._logger.debug(
            "off barcode cache miss — calling API",
            barcode=barcode,
        )

        url = f"{self._base_url}/api/v2/product/{barcode}.json"
        headers = {"User-Agent": self._user_agent}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=headers)
        except Exception:
            self._logger.error(
                "off barcode lookup failed",
                barcode=barcode,
            )
            raise

        if response.status_code == 404:
            self._logger.info(
                "off barcode not found (404) — caching miss",
                barcode=barcode,
            )
            self._cache.set(cache_key, _NOT_FOUND_SENTINEL, ttl_seconds=self._ttl)
            return None

        response.raise_for_status()
        data = response.json()

        # OFF returns status=0 for invalid/unknown barcodes on 200 responses.
        if data.get("status") == 0:
            self._logger.info(
                "off barcode status=0 — product not found",
                barcode=barcode,
            )
            self._cache.set(cache_key, _NOT_FOUND_SENTINEL, ttl_seconds=self._ttl)
            return None

        raw_product = data.get("product") or {}
        product = _product_from_raw({**raw_product, "code": barcode})

        self._cache.set(cache_key, product.model_dump(), ttl_seconds=self._ttl)
        self._logger.info(
            "off barcode lookup complete",
            barcode=barcode,
            product_name=product.product_name,
        )
        return product

    async def search_by_name(
        self,
        name: str,
        brand: str | None = None,
        page_size: int = 10,
    ) -> list[OFFProduct]:
        """Search OFF by product name and optional brand.

        Args:
            name: Product name (e.g., ``"Total 0% Greek Yogurt"``).
            brand: Optional brand filter (e.g., ``"Fage"``).
            page_size: Maximum results to return.

        Returns:
            Ranked :class:`OFFProduct` objects — may be empty.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
            httpx.TransportError: On network-level failures.
        """
        brand_lower = (brand or "").lower()
        name_lower = name.lower()
        cache_key = f"off:search:{name_lower}:{brand_lower}:{page_size}:v1"

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._logger.debug(
                "off search cache hit",
                name=name,
                brand=brand,
                cache_key=cache_key,
            )
            return [OFFProduct(**item) for item in cached]

        self._logger.debug(
            "off search cache miss — calling API",
            name=name,
            brand=brand,
            page_size=page_size,
        )

        url = f"{self._base_url}/cgi/search.pl"
        params: dict[str, str | int] = {
            "search_terms": name,
            "json": "1",
            "page_size": page_size,
        }
        if brand:
            params["brands"] = brand

        headers = {"User-Agent": self._user_agent}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params, headers=headers)
        except Exception:
            self._logger.error(
                "off search failed",
                name=name,
                brand=brand,
            )
            raise

        response.raise_for_status()
        data = response.json()

        raw_products: list[dict[str, Any]] = data.get("products") or []
        products = [_product_from_raw(r) for r in raw_products]

        self._cache.set(
            cache_key,
            [p.model_dump() for p in products],
            ttl_seconds=self._ttl,
        )
        self._logger.info(
            "off search complete",
            name=name,
            brand=brand,
            result_count=len(products),
        )
        return products
