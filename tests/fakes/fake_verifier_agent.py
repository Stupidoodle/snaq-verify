"""FakeVerifierAgent — in-memory VerifierAgentPort for unit tests."""

from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort
from tests.fakes.factories import make_item_verification


class FakeVerifierAgent(VerifierAgentPort):
    """Returns configurable ItemVerification results without any LLM calls.

    By default, each call to `verify` returns a ``MATCH`` verification built
    from the supplied item.  Tests can override the result with `fixed_result`
    or cause a named item to raise via `raise_for_id`.
    """

    def __init__(
        self,
        fixed_result: ItemVerification | None = None,
        raise_for_id: str | None = None,
    ) -> None:
        self._fixed_result = fixed_result
        self._raise_for_id = raise_for_id
        self.calls: list[FoodItem] = []

    async def verify(self, item: FoodItem) -> ItemVerification:
        """Return a pre-configured or generated ItemVerification."""
        self.calls.append(item)
        if self._raise_for_id is not None and item.id == self._raise_for_id:
            raise RuntimeError(f"FakeVerifierAgent: simulated error for '{item.id}'")
        if self._fixed_result is not None:
            return self._fixed_result
        return make_item_verification(item=item)
