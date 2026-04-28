"""FakeVerifierAgent — in-memory VerifierAgentPort for unit tests."""

from __future__ import annotations

from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort
from tests.fakes.factories import make_item_verification


class FakeVerifierAgent(VerifierAgentPort):
    """Returns configurable ItemVerification results without any LLM calls.

    By default, each call to ``verify`` returns a ``MATCH`` verification built
    from the supplied item.  Tests can override the result with ``fixed_result``
    or cause a named item to raise via ``raise_for_id``.

    The ``hint`` parameter added for ``SelfVerifyStep`` is accepted but
    intentionally ignored — fakes do not perform real re-verification.
    """

    def __init__(
        self,
        fixed_result: ItemVerification | None = None,
        raise_for_id: str | None = None,
    ) -> None:
        self._fixed_result = fixed_result
        self._raise_for_id = raise_for_id
        self.calls: list[FoodItem] = []
        self.hints: list[str | None] = []

    async def verify(
        self, item: FoodItem, hint: str | None = None,
    ) -> ItemVerification:
        """Return a pre-configured or generated ItemVerification.

        Args:
            item: The food item to (fake-)verify.
            hint: Optional hint from a prior judge run.  Recorded in
                ``self.hints`` for assertions in tests; otherwise ignored.
        """
        self.calls.append(item)
        self.hints.append(hint)
        if self._raise_for_id is not None and item.id == self._raise_for_id:
            raise RuntimeError(f"FakeVerifierAgent: simulated error for '{item.id}'")
        if self._fixed_result is not None:
            return self._fixed_result
        return make_item_verification(item=item)
