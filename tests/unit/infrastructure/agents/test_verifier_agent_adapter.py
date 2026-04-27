"""Unit tests for VerifierAgentAdapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import set_tracing_disabled

from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.infrastructure.agents.verifier_agent_adapter import (
    VerifierAgentAdapter,
)
from tests.fakes.factories import make_food_item, make_item_verification
from tests.fakes.fake_logger import FakeLogger

set_tracing_disabled(True)

_RUNNER_RUN = (
    "snaq_verify.infrastructure.agents.verifier_agent_adapter.Runner.run"
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SETTINGS_STUB = MagicMock()
_SETTINGS_STUB.OPENAI_MODEL = "gpt-test"
_SETTINGS_STUB.MATCH_TOLERANCE_PCT = 5.0
_SETTINGS_STUB.MINOR_TOLERANCE_PCT = 15.0
_SETTINGS_STUB.ATWATER_TOLERANCE_PCT = 15.0
_SETTINGS_STUB.MIN_CANDIDATE_SCORE = 0.5
_SETTINGS_STUB.ABSOLUTE_FLOOR_G = 0.5


def _make_adapter() -> VerifierAgentAdapter:
    return VerifierAgentAdapter(
        settings=_SETTINGS_STUB,
        logger=FakeLogger(),
        usda=MagicMock(),
        off=MagicMock(),
        tavily=MagicMock(),
    )


def _make_run_result(verification: ItemVerification) -> MagicMock:
    """Return a fake RunResult whose final_output_as returns the given verification."""
    result = MagicMock()
    result.final_output_as.return_value = verification
    return result


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_adapter_construction_succeeds() -> None:
    """VerifierAgentAdapter can be constructed without errors."""
    adapter = _make_adapter()
    assert isinstance(adapter, VerifierAgentAdapter)


def test_adapter_implements_port() -> None:
    """VerifierAgentAdapter satisfies the VerifierAgentPort ABC."""
    from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort

    adapter = _make_adapter()
    assert isinstance(adapter, VerifierAgentPort)


# ---------------------------------------------------------------------------
# verify() — happy path
# ---------------------------------------------------------------------------


async def test_verify_returns_item_verification() -> None:
    """verify() forwards the runner's output as an ItemVerification."""
    item = make_food_item()
    expected = make_item_verification(item=item)
    run_result = _make_run_result(expected)

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert result is expected


async def test_verify_passes_item_json_in_prompt() -> None:
    """verify() includes the item's JSON representation in the runner input."""
    item = make_food_item(item_id="banana-raw", name="Banana, Raw")
    run_result = _make_run_result(make_item_verification(item=item))

    captured_input: list[str] = []

    async def _capture_run(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        captured_input.append(input)
        return run_result

    with patch(_RUNNER_RUN, new=_capture_run):
        adapter = _make_adapter()
        await adapter.verify(item)

    assert len(captured_input) == 1
    assert "banana-raw" in captured_input[0]
    assert "Banana" in captured_input[0]


async def test_verify_creates_context_with_ports() -> None:
    """verify() builds a VerifierContext that carries the injected ports."""
    from snaq_verify.infrastructure.agents.verifier_agent import VerifierContext

    item = make_food_item()
    run_result = _make_run_result(make_item_verification(item=item))

    captured_ctx: list[VerifierContext] = []

    async def _capture_run(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        captured_ctx.append(context)
        return run_result

    usda_mock = MagicMock()
    off_mock = MagicMock()
    tavily_mock = MagicMock()

    with patch(_RUNNER_RUN, new=_capture_run):
        adapter = VerifierAgentAdapter(
            settings=_SETTINGS_STUB,
            logger=FakeLogger(),
            usda=usda_mock,
            off=off_mock,
            tavily=tavily_mock,
        )
        await adapter.verify(item)

    assert len(captured_ctx) == 1
    ctx = captured_ctx[0]
    assert isinstance(ctx, VerifierContext)
    assert ctx.usda is usda_mock
    assert ctx.off is off_mock
    assert ctx.tavily is tavily_mock


async def test_verify_logs_start_and_done() -> None:
    """verify() emits info-level logs at the start and end."""
    item = make_food_item(item_id="egg-whole-raw")
    run_result = _make_run_result(make_item_verification(item=item))
    logger = FakeLogger()

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = VerifierAgentAdapter(
            settings=_SETTINGS_STUB,
            logger=logger,
            usda=MagicMock(),
            off=MagicMock(),
            tavily=MagicMock(),
        )
        await adapter.verify(item)

    messages = [m[1] for m in logger.messages]
    assert any("start" in m for m in messages)
    assert any("done" in m for m in messages)


# ---------------------------------------------------------------------------
# verify() — error propagation
# ---------------------------------------------------------------------------


async def test_verify_propagates_runner_error() -> None:
    """RuntimeError from Runner.run propagates unchanged."""
    item = make_food_item()

    async def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated runner error")

    with patch(_RUNNER_RUN, new=_raise):
        adapter = _make_adapter()
        with pytest.raises(RuntimeError, match="simulated runner error"):
            await adapter.verify(item)
