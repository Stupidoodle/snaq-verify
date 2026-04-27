"""Unit tests for JudgeAgentAdapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import set_tracing_disabled

from snaq_verify.domain.models.eval_models import JudgeVerdict
from snaq_verify.infrastructure.agents.judge_agent_adapter import JudgeAgentAdapter
from tests.fakes.factories import (
    make_ground_truth_entry,
    make_item_verification,
    make_judge_verdict,
)
from tests.fakes.fake_logger import FakeLogger

set_tracing_disabled(True)

_RUNNER_RUN = "snaq_verify.infrastructure.agents.judge_agent_adapter.Runner.run"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SETTINGS_STUB = MagicMock()
_SETTINGS_STUB.OPENAI_MODEL = "gpt-test"


def _make_adapter() -> JudgeAgentAdapter:
    return JudgeAgentAdapter(settings=_SETTINGS_STUB, logger=FakeLogger())


def _make_run_result(verdict: JudgeVerdict) -> MagicMock:
    result = MagicMock()
    result.final_output_as.return_value = verdict
    return result


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_adapter_construction_succeeds() -> None:
    """JudgeAgentAdapter can be constructed without errors."""
    adapter = _make_adapter()
    assert isinstance(adapter, JudgeAgentAdapter)


def test_adapter_implements_port() -> None:
    """JudgeAgentAdapter satisfies the JudgeAgentPort ABC."""
    from snaq_verify.domain.ports.judge_agent_port import JudgeAgentPort

    adapter = _make_adapter()
    assert isinstance(adapter, JudgeAgentPort)


# ---------------------------------------------------------------------------
# judge() — happy path
# ---------------------------------------------------------------------------


async def test_judge_returns_judge_verdict() -> None:
    """judge() forwards the runner's output as a JudgeVerdict."""
    verification = make_item_verification()
    ground_truth = make_ground_truth_entry()
    expected = make_judge_verdict(
        item_id=verification.item_id, score=0.9, correct_verdict=True,
    )
    run_result = _make_run_result(expected)

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.judge(verification, ground_truth)

    assert result is expected


async def test_judge_prompt_contains_both_jsons() -> None:
    """judge() includes both verification and ground-truth JSON in the prompt."""
    verification = make_item_verification()
    ground_truth = make_ground_truth_entry(item_id=verification.item_id)
    run_result = _make_run_result(make_judge_verdict(item_id=verification.item_id))

    captured_input: list[str] = []

    async def _capture_run(agent, input, **kwargs):  # type: ignore[no-untyped-def]
        captured_input.append(input)
        return run_result

    with patch(_RUNNER_RUN, new=_capture_run):
        adapter = _make_adapter()
        await adapter.judge(verification, ground_truth)

    assert len(captured_input) == 1
    prompt = captured_input[0]
    # Both verification and ground-truth data must appear in the prompt
    assert verification.item_id in prompt
    assert ground_truth.source_url in prompt


async def test_judge_logs_start_and_done() -> None:
    """judge() emits info-level log messages at start and completion."""
    verification = make_item_verification()
    ground_truth = make_ground_truth_entry()
    run_result = _make_run_result(make_judge_verdict(item_id=verification.item_id))
    logger = FakeLogger()

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = JudgeAgentAdapter(settings=_SETTINGS_STUB, logger=logger)
        await adapter.judge(verification, ground_truth)

    messages = [m[1] for m in logger.messages]
    assert any("start" in m for m in messages)
    assert any("done" in m for m in messages)


async def test_judge_no_context_passed_to_runner() -> None:
    """judge() calls Runner.run without a context (judge agent has no IO tools)."""
    verification = make_item_verification()
    ground_truth = make_ground_truth_entry()
    run_result = _make_run_result(make_judge_verdict(item_id=verification.item_id))

    captured_kwargs: list[dict] = []

    async def _capture_run(agent, input, **kwargs):  # type: ignore[no-untyped-def]
        captured_kwargs.append(kwargs)
        return run_result

    with patch(_RUNNER_RUN, new=_capture_run):
        adapter = _make_adapter()
        await adapter.judge(verification, ground_truth)

    # No context= kwarg (or context is None/absent)
    assert len(captured_kwargs) == 1
    kw = captured_kwargs[0]
    assert kw.get("context") is None or "context" not in kw


# ---------------------------------------------------------------------------
# judge() — error propagation
# ---------------------------------------------------------------------------


async def test_judge_propagates_runner_error() -> None:
    """RuntimeError from Runner.run propagates unchanged."""
    verification = make_item_verification()
    ground_truth = make_ground_truth_entry()

    async def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated judge error")

    with patch(_RUNNER_RUN, new=_raise):
        adapter = _make_adapter()
        with pytest.raises(RuntimeError, match="simulated judge error"):
            await adapter.judge(verification, ground_truth)
