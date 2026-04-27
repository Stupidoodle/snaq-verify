"""USDA FoodData Central client port."""

from abc import ABC, abstractmethod

from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.models.source_lookup import USDACandidate


class USDAClientPort(ABC):
    """Abstract interface for the USDA FoodData Central API client.

    Two operations: `search` (ranked candidates by query) and `get_food`
    (full nutrition for a single fdcId). Adapters return `USDACandidate`
    objects with `nutrition_per_100g` populated from the FDC nutrient list.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        data_type: USDADataType | None = None,
        page_size: int = 10,
    ) -> list[USDACandidate]:
        """Search the USDA database for candidates matching `query`.

        Args:
            query: Free-text food name.
            data_type: Optional filter (Foundation/SR Legacy/Branded/Survey).
                When None, no filter is applied — caller picks data type
                priority via downstream scoring.
            page_size: Max results to return (FDC supports 1–200).

        Returns:
            Ranked candidates. Empty list when no matches. Raises on
            transport / auth errors.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_food(self, fdc_id: int) -> USDACandidate:
        """Fetch the full nutrition payload for a specific fdcId.

        Args:
            fdc_id: USDA food identifier.

        Returns:
            A `USDACandidate` with `nutrition_per_100g` populated. Raises
            on 404 or transport errors.
        """
        raise NotImplementedError
