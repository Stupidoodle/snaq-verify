"""Unit tests for VerifyStep."""

import pytest

from snaq_verify.application.pipeline.steps.verify_step import VerifyStep
from snaq_verify.domain.models.enums import Verdict
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.factories import make_food_item, make_item_verification
from tests.fakes.fake_logger import FakeLogger
from tests.fakes.fake_verifier_agent import FakeVerifierAgent


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_verifies_single_item() -> None:
    """Single item is verified and placed in state.verifications."""
    item = make_food_item()
    agent = FakeVerifierAgent()

    step = VerifyStep(verifier_agent=agent, logger=FakeLogger())
    state = PipelineState(items=[item])
    result = await step.run(state)

    assert len(result.verifications) == 1
    assert result.verifications[0].item_id == item.id
    assert agent.calls == [item]


async def test_verifications_preserve_input_order() -> None:
    """Output order mirrors input order regardless of concurrency."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(6)]
    agent = FakeVerifierAgent()

    step = VerifyStep(verifier_agent=agent, logger=FakeLogger(), concurrency=3)
    result = await step.run(PipelineState(items=items))

    assert [v.item_id for v in result.verifications] == [f"item-{i}" for i in range(6)]


async def test_empty_items_yields_empty_verifications() -> None:
    """Empty input list yields empty verifications without calling the agent."""
    agent = FakeVerifierAgent()
    step = VerifyStep(verifier_agent=agent, logger=FakeLogger())
    result = await step.run(PipelineState(items=[]))

    assert result.verifications == []
    assert agent.calls == []


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert VerifyStep(
        verifier_agent=FakeVerifierAgent(), logger=FakeLogger()
    ).name == "verify"


async def test_on_item_complete_callback_called_for_each_item() -> None:
    """on_item_complete is called once per item."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(3)]
    completed: list[str] = []

    step = VerifyStep(
        verifier_agent=FakeVerifierAgent(),
        logger=FakeLogger(),
        on_item_complete=completed.append,
    )
    await step.run(PipelineState(items=items))

    assert sorted(completed) == ["item-0", "item-1", "item-2"]


async def test_uses_fixed_result_from_agent() -> None:
    """Fixed result from agent is stored verbatim."""
    item = make_food_item()
    fixed = make_item_verification(item=item, verdict=Verdict.MAJOR_DISCREPANCY)
    agent = FakeVerifierAgent(fixed_result=fixed)

    step = VerifyStep(verifier_agent=agent, logger=FakeLogger())
    result = await step.run(PipelineState(items=[item]))

    assert result.verifications[0].verdict == Verdict.MAJOR_DISCREPANCY


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


async def test_agent_error_produces_fallback_and_continues() -> None:
    """A RuntimeError from the agent is caught — the failing item gets a
    NO_DATA fallback verification with an `agent_failure_agent_error` note,
    and the remaining items in the batch still complete. Aborting the whole
    11-item run because one item flapped is the wrong default for a
    verification system.
    """
    items = [make_food_item(item_id="bad"), make_food_item(item_id="good")]
    agent = FakeVerifierAgent(raise_for_id="bad")

    step = VerifyStep(verifier_agent=agent, logger=FakeLogger(), concurrency=1)
    result = await step.run(PipelineState(items=items))

    assert len(result.verifications) == 2
    bad, good = result.verifications
    assert bad.item_id == "bad"
    assert bad.verdict == Verdict.NO_DATA
    assert any("agent_failure_agent_error" in n for n in bad.notes)
    assert good.item_id == "good"
    assert good.verdict != Verdict.NO_DATA  # the unaffected item was processed normally


# ---------------------------------------------------------------------------
# Concurrency / semaphore
# ---------------------------------------------------------------------------


async def test_concurrency_cap_respected() -> None:
    """Items are verified with at most `concurrency` simultaneous calls.

    We test this indirectly: 10 items with concurrency=2 complete without
    error and return all 10 verifications.
    """
    items = [make_food_item(item_id=f"item-{i}") for i in range(10)]
    step = VerifyStep(
        verifier_agent=FakeVerifierAgent(), logger=FakeLogger(), concurrency=2
    )
    result = await step.run(PipelineState(items=items))

    assert len(result.verifications) == 10
