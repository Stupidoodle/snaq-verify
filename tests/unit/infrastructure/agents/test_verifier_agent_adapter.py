"""Unit tests for VerifierAgentAdapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import ReasoningItem, set_tracing_disabled
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

from snaq_verify.domain.models.enums import ConfidenceLevel
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
    """verify() returns an ItemVerification derived from the runner's output."""
    item = make_food_item()
    # make_item_verification produces 1 source with match_score=0.95 → MEDIUM
    expected = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)
    run_result = _make_run_result(expected)

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert isinstance(result, ItemVerification)
    assert result.item_id == expected.item_id
    assert result.item_name == expected.item_name


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
    assert ctx.item is item  # item passed for is_enabled barcode gate


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
# verify() — post-run overrides
# ---------------------------------------------------------------------------


async def test_verify_overrides_notes_with_tool_events() -> None:
    """verify() replaces LLM notes with context.tool_events after run."""
    from snaq_verify.infrastructure.agents.verifier_agent import VerifierContext

    item = make_food_item()
    # Agent returns a verification with stale fabricated notes
    base = make_item_verification(item=item)
    base = base.model_copy(update={"notes": ["fabricated error message"]})
    run_result = _make_run_result(base)

    captured_ctx: list[VerifierContext] = []

    async def _capture_and_populate_run(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        captured_ctx.append(context)
        # Simulate a tool event that the IO tool would have appended
        context.tool_events.append("off.lookup_by_barcode.not_found barcode='12345'")
        return run_result

    with patch(_RUNNER_RUN, new=_capture_and_populate_run):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    # Notes must come from tool_events, not from the LLM's output
    assert result.notes == ["off.lookup_by_barcode.not_found barcode='12345'"]


async def test_verify_notes_empty_when_no_tool_events() -> None:
    """verify() sets notes=[] when context.tool_events is empty."""
    item = make_food_item()
    base = make_item_verification(item=item)
    # Even if LLM wrote something, tool_events=[] wins
    base_with_notes = base.model_copy(update={"notes": ["llm wrote this"]})
    run_result = _make_run_result(base_with_notes)

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert result.notes == []


async def test_verify_overrides_confidence_to_deterministic_value() -> None:
    """verify() corrects the LLM's confidence to the deterministic derived value.

    make_item_verification produces 1 evidence source with match_score=0.95.
    derive_confidence: 1 source, score ≥ 0.70 → MEDIUM.
    If the LLM returned HIGH, the adapter must correct it to MEDIUM.
    """
    item = make_food_item()
    # LLM incorrectly says HIGH (only 1 source in evidence → should be MEDIUM)
    wrong = make_item_verification(item=item, confidence=ConfidenceLevel.HIGH)
    run_result = _make_run_result(wrong)

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert result.confidence == ConfidenceLevel.MEDIUM


async def test_verify_no_override_when_confidence_already_correct() -> None:
    """verify() does not mutate the output when confidence is already correct."""
    item = make_food_item()
    # 1 source with match_score=0.95 → MEDIUM
    correct = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)
    run_result = _make_run_result(correct)

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert result.confidence == ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# verify() — reasoning audit trail
# ---------------------------------------------------------------------------


def _make_reasoning_item(texts: list[str]) -> ReasoningItem:
    """Build a fake ReasoningItem with the given summary texts."""
    raw = ResponseReasoningItem(
        id="test-reasoning-01",
        type="reasoning",
        summary=[Summary(text=t, type="summary_text") for t in texts],
    )
    return ReasoningItem(agent=MagicMock(), raw_item=raw)


async def test_verify_keeps_llm_reasoning_when_no_native_items() -> None:
    """LLM's self-reported reasoning is kept when Runner has no ReasoningItems."""
    item = make_food_item()
    base = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)
    base = base.model_copy(update={"reasoning": "Called USDA; candidate match 0.92."})
    run_result = _make_run_result(base)
    run_result.new_items = []  # no native reasoning items

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert result.reasoning == "Called USDA; candidate match 0.92."


async def test_verify_overrides_reasoning_with_native_items() -> None:
    """Adapter replaces LLM reasoning with native ReasoningItem text."""
    item = make_food_item()
    base = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)
    base = base.model_copy(update={"reasoning": "LLM self-report (should be replaced)"})
    run_result = _make_run_result(base)
    run_result.new_items = [
        _make_reasoning_item(
            ["Called search_usda(Foundation).", "Candidate match_score=0.89."],
        ),
    ]

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    expected = "Called search_usda(Foundation).\n\nCandidate match_score=0.89."
    assert result.reasoning == expected


async def test_verify_multiple_reasoning_items_joined() -> None:
    """Multiple ReasoningItems' summary texts are all joined with double newlines."""
    item = make_food_item()
    base = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)
    run_result = _make_run_result(base)
    run_result.new_items = [
        _make_reasoning_item(["First reasoning block."]),
        _make_reasoning_item(["Second reasoning block."]),
    ]

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert "First reasoning block." in result.reasoning  # type: ignore[operator]
    assert "Second reasoning block." in result.reasoning  # type: ignore[operator]


async def test_verify_reasoning_stays_none_when_no_items_and_no_llm_reasoning() -> None:
    """reasoning stays None: no native items and LLM did not populate it."""
    item = make_food_item()
    base = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)
    assert base.reasoning is None
    run_result = _make_run_result(base)
    run_result.new_items = []

    with patch(_RUNNER_RUN, new=AsyncMock(return_value=run_result)):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert result.reasoning is None


# ---------------------------------------------------------------------------
# verify() — is_enabled barcode gate (VerifierContext.item)
# ---------------------------------------------------------------------------


async def test_verify_context_carries_item() -> None:
    """VerifierContext.item is the exact FoodItem passed to verify()."""
    from snaq_verify.infrastructure.agents.verifier_agent import VerifierContext

    item = make_food_item(item_id="fage-total-0-greek-yogurt")
    run_result = _make_run_result(make_item_verification(item=item))

    captured_ctx: list[VerifierContext] = []

    async def _capture_run(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        captured_ctx.append(context)
        return run_result

    with patch(_RUNNER_RUN, new=_capture_run):
        adapter = _make_adapter()
        await adapter.verify(item)

    assert captured_ctx[0].item is item


# ---------------------------------------------------------------------------
# verify() — low-confidence retry loop
# ---------------------------------------------------------------------------


async def test_verify_no_retry_when_confidence_not_low() -> None:
    """Runner.run is called exactly once when confidence is MEDIUM or higher."""
    item = make_food_item()
    # MEDIUM confidence (1 source, match_score=0.95 → MEDIUM by derive_confidence)
    verification = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)
    run_result = _make_run_result(verification)

    call_count = 0

    async def _count_calls(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return run_result

    with patch(_RUNNER_RUN, new=_count_calls):
        adapter = _make_adapter()
        await adapter.verify(item)

    assert call_count == 1


async def test_verify_retries_once_on_low_confidence() -> None:
    """Runner.run is called twice when the first result has LOW confidence."""
    item = make_food_item()
    # First run → LOW confidence (0 evidence sources → derive_confidence=LOW)
    low_v = make_item_verification(item=item, confidence=ConfidenceLevel.LOW)
    low_v = low_v.model_copy(update={"evidence": []})

    # Second run → MEDIUM confidence (evidence restored)
    medium_v = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)

    call_count = 0
    inputs_received: list[object] = []

    def _make_low_run() -> MagicMock:
        r = MagicMock()
        r.final_output_as.return_value = low_v
        r.new_items = []
        r.to_input_list.return_value = [{"role": "assistant", "content": "prior turn"}]
        return r

    def _make_medium_run() -> MagicMock:
        r = MagicMock()
        r.final_output_as.return_value = medium_v
        r.new_items = []
        return r

    async def _two_phase_run(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        inputs_received.append(input)
        return _make_low_run() if call_count == 1 else _make_medium_run()

    with patch(_RUNNER_RUN, new=_two_phase_run):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    assert call_count == 2
    # Second call must include the retry feedback prompt
    retry_input = inputs_received[1]
    assert isinstance(retry_input, list)
    assert any("LOW" in str(msg) for msg in retry_input)
    # Final result should reflect the second run's output
    assert result.item_id == medium_v.item_id


async def test_verify_accepts_low_confidence_after_retry() -> None:
    """Retry still returns LOW → adapter accepts it (no infinite loop)."""
    item = make_food_item()
    low_v = make_item_verification(item=item, confidence=ConfidenceLevel.LOW)
    low_v = low_v.model_copy(update={"evidence": []})

    call_count = 0

    def _make_low_run() -> MagicMock:
        r = MagicMock()
        r.final_output_as.return_value = low_v
        r.new_items = []
        r.to_input_list.return_value = []
        return r

    async def _always_low(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return _make_low_run()

    with patch(_RUNNER_RUN, new=_always_low):
        adapter = _make_adapter()
        result = await adapter.verify(item)

    # Maximum 2 calls — no more
    assert call_count == 2
    assert isinstance(result, ItemVerification)


async def test_verify_retry_logs_retry_event() -> None:
    """verify() logs a retry event when confidence is LOW."""
    item = make_food_item()
    low_v = make_item_verification(item=item, confidence=ConfidenceLevel.LOW)
    low_v = low_v.model_copy(update={"evidence": []})
    medium_v = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)

    call_count = 0

    def _make_low_run() -> MagicMock:
        r = MagicMock()
        r.final_output_as.return_value = low_v
        r.new_items = []
        r.to_input_list.return_value = []
        return r

    def _make_medium_run() -> MagicMock:
        r = MagicMock()
        r.final_output_as.return_value = medium_v
        r.new_items = []
        return r

    async def _two_phase(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return _make_low_run() if call_count == 1 else _make_medium_run()

    logger = FakeLogger()
    with patch(_RUNNER_RUN, new=_two_phase):
        adapter = VerifierAgentAdapter(
            settings=_SETTINGS_STUB,
            logger=logger,
            usda=MagicMock(),
            off=MagicMock(),
            tavily=MagicMock(),
        )
        await adapter.verify(item)

    messages = [m[1] for m in logger.messages]
    assert any("retry" in m for m in messages)


# ---------------------------------------------------------------------------
# verify() — max_turns
# ---------------------------------------------------------------------------


async def test_verify_passes_max_turns_to_runner() -> None:
    """verify() passes max_turns=30 to Runner.run."""
    item = make_food_item()
    run_result = _make_run_result(
        make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM),
    )

    captured_kwargs: list[dict] = []  # type: ignore[type-arg]

    async def _capture_run(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        captured_kwargs.append(kwargs)
        return run_result

    with patch(_RUNNER_RUN, new=_capture_run):
        adapter = _make_adapter()
        await adapter.verify(item)

    assert captured_kwargs[0].get("max_turns") == 30


async def test_verify_passes_max_turns_on_retry_run() -> None:
    """max_turns=30 is also applied to the low-confidence retry Runner.run call."""
    item = make_food_item()
    low_v = make_item_verification(item=item, confidence=ConfidenceLevel.LOW)
    low_v = low_v.model_copy(update={"evidence": []})
    medium_v = make_item_verification(item=item, confidence=ConfidenceLevel.MEDIUM)

    call_count = 0
    all_kwargs: list[dict] = []  # type: ignore[type-arg]

    def _make_low_run() -> MagicMock:
        r = MagicMock()
        r.final_output_as.return_value = low_v
        r.new_items = []
        r.to_input_list.return_value = []
        return r

    def _make_medium_run() -> MagicMock:
        r = MagicMock()
        r.final_output_as.return_value = medium_v
        r.new_items = []
        return r

    async def _two_phase(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        all_kwargs.append(kwargs)
        return _make_low_run() if call_count == 1 else _make_medium_run()

    with patch(_RUNNER_RUN, new=_two_phase):
        adapter = _make_adapter()
        await adapter.verify(item)

    assert call_count == 2
    assert all(kw.get("max_turns") == 30 for kw in all_kwargs)


# ---------------------------------------------------------------------------
# verify() — guardrail trip diagnostics
# ---------------------------------------------------------------------------


async def test_verify_logs_guardrail_trip_and_reraises() -> None:
    """verify() logs guardrail name at WARNING then re-raises tripwire exception."""
    from agents import OutputGuardrailTripwireTriggered

    item = make_food_item()

    # Build a minimal fake OutputGuardrailTripwireTriggered
    guardrail_mock = MagicMock()
    guardrail_mock.get_name.return_value = "atwater_output_guardrail"
    output_mock = MagicMock()
    output_mock.output_info = {"delta_pct": 25.0}
    guardrail_result_mock = MagicMock()
    guardrail_result_mock.guardrail = guardrail_mock
    guardrail_result_mock.output = output_mock

    exc = OutputGuardrailTripwireTriggered(guardrail_result_mock)

    async def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise exc

    logger = FakeLogger()
    with patch(_RUNNER_RUN, new=_raise):
        adapter = VerifierAgentAdapter(
            settings=_SETTINGS_STUB,
            logger=logger,
            usda=MagicMock(),
            off=MagicMock(),
            tavily=MagicMock(),
        )
        with pytest.raises(OutputGuardrailTripwireTriggered):
            await adapter.verify(item)

    trip_entries = [
        (lvl, msg, kw)
        for lvl, msg, kw in logger.messages
        if msg == "verifier_adapter.guardrail_trip"
    ]
    assert len(trip_entries) == 1
    lvl, _msg, kw = trip_entries[0]
    assert lvl == "warning"
    assert kw["guardrail"] == "atwater_output_guardrail"
    assert kw["item_id"] == item.id


async def test_verify_guardrail_trip_appends_tool_event() -> None:
    """verify() appends an agent_failure_guardrail_trip note to context.tool_events."""
    from agents import OutputGuardrailTripwireTriggered

    from snaq_verify.infrastructure.agents.verifier_agent import VerifierContext

    item = make_food_item()

    guardrail_mock = MagicMock()
    guardrail_mock.get_name.return_value = "schema_output_guardrail"
    output_mock = MagicMock()
    output_mock.output_info = {"schema_issues": ["proposed_correction missing fields"]}
    guardrail_result_mock = MagicMock()
    guardrail_result_mock.guardrail = guardrail_mock
    guardrail_result_mock.output = output_mock

    exc = OutputGuardrailTripwireTriggered(guardrail_result_mock)
    captured_ctx: list[VerifierContext] = []

    async def _capture_and_raise(agent, input, *, context, **kwargs):  # type: ignore[no-untyped-def]
        captured_ctx.append(context)
        raise exc

    with patch(_RUNNER_RUN, new=_capture_and_raise):
        adapter = _make_adapter()
        with pytest.raises(OutputGuardrailTripwireTriggered):
            await adapter.verify(item)

    assert len(captured_ctx) == 1
    events = captured_ctx[0].tool_events
    assert any("agent_failure_guardrail_trip" in e for e in events)
    assert any("schema_output_guardrail" in e for e in events)


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
