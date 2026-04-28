"""Unit tests for SelfVerifyStep."""

from snaq_verify.application.pipeline.steps.self_verify_step import SelfVerifyStep
from snaq_verify.domain.models.enums import ConfidenceLevel, Verdict
from snaq_verify.domain.models.eval_models import JudgeVerdict
from tests.fakes.factories import (
    make_food_item,
    make_ground_truth_entry,
    make_item_verification,
    make_pipeline_state,
)
from tests.fakes.fake_judge_agent import FakeJudgeAgent
from tests.fakes.fake_logger import FakeLogger
from tests.fakes.fake_verifier_agent import FakeVerifierAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    verifier: FakeVerifierAgent | None = None,
    judge: FakeJudgeAgent | None = None,
    threshold: float = 0.5,
) -> SelfVerifyStep:
    return SelfVerifyStep(
        verifier_agent=verifier or FakeVerifierAgent(),
        judge_agent=judge or FakeJudgeAgent(),
        logger=FakeLogger(),
        retry_threshold=threshold,
    )


def _low_verdict(item_id: str, score: float = 0.3) -> JudgeVerdict:
    return JudgeVerdict(
        item_id=item_id,
        score=score,
        correct_verdict=False,
        reasoning="Evidence too thin; fat value not verified.",
    )


def _high_verdict(item_id: str, score: float = 0.9) -> JudgeVerdict:
    return JudgeVerdict(
        item_id=item_id,
        score=score,
        correct_verdict=True,
        reasoning="Correct verdict with strong USDA evidence.",
    )


# ---------------------------------------------------------------------------
# Step metadata
# ---------------------------------------------------------------------------


def test_self_verify_step_name() -> None:
    """SelfVerifyStep.name returns 'self_verify'."""
    assert _make_step().name == "self_verify"


# ---------------------------------------------------------------------------
# Skipping behaviour
# ---------------------------------------------------------------------------


async def test_skips_when_no_verifications() -> None:
    """Step returns early when state.verifications is empty."""
    judge = FakeJudgeAgent()
    step = _make_step(judge=judge)
    state = make_pipeline_state(items=[], verifications=[], ground_truth=[])

    result = await step.run(state)

    assert result.verifications == []
    assert judge.calls == []  # no judge calls made


async def test_skips_item_without_ground_truth() -> None:
    """Items with no matching ground-truth entry are returned unchanged."""
    item = make_food_item(item_id="no-gt-item")
    verification = make_item_verification(item=item)
    judge = FakeJudgeAgent()
    step = _make_step(judge=judge)

    state = make_pipeline_state(
        items=[item],
        verifications=[verification],
        ground_truth=[],  # no GT entries
    )
    result = await step.run(state)

    assert result.verifications[0] is verification
    assert judge.calls == []


# ---------------------------------------------------------------------------
# Good score — keep original
# ---------------------------------------------------------------------------


async def test_keeps_verification_above_threshold() -> None:
    """Judge score ≥ threshold → original verification is kept unchanged."""
    item = make_food_item(item_id="avocado-raw")
    gt = make_ground_truth_entry(item_id="avocado-raw")
    verification = make_item_verification(item=item)
    verifier = FakeVerifierAgent()
    judge = FakeJudgeAgent(
        results_by_id={"avocado-raw": _high_verdict("avocado-raw", score=0.9)},
    )
    step = _make_step(verifier=verifier, judge=judge)

    state = make_pipeline_state(
        items=[item], verifications=[verification], ground_truth=[gt],
    )
    result = await step.run(state)

    assert result.verifications[0] is verification  # unchanged
    assert verifier.calls == []  # no re-verification


async def test_keeps_verification_at_exact_threshold() -> None:
    """Judge score == threshold exactly → keep original (boundary inclusive)."""
    item = make_food_item(item_id="egg-raw")
    gt = make_ground_truth_entry(item_id="egg-raw")
    verification = make_item_verification(item=item)
    verifier = FakeVerifierAgent()
    judge = FakeJudgeAgent(
        results_by_id={"egg-raw": _high_verdict("egg-raw", score=0.5)},
    )
    step = _make_step(verifier=verifier, judge=judge, threshold=0.5)

    state = make_pipeline_state(
        items=[item], verifications=[verification], ground_truth=[gt],
    )
    result = await step.run(state)

    assert result.verifications[0] is verification
    assert verifier.calls == []


# ---------------------------------------------------------------------------
# Low score — retry with hint
# ---------------------------------------------------------------------------


async def test_retries_low_score_item() -> None:
    """Judge score < threshold → verifier is called with hint."""
    item = make_food_item(item_id="chicken-breast-raw")
    gt = make_ground_truth_entry(item_id="chicken-breast-raw")
    original = make_item_verification(item=item, verdict=Verdict.MATCH)
    improved = make_item_verification(
        item=item, verdict=Verdict.MAJOR_DISCREPANCY,
        confidence=ConfidenceLevel.MEDIUM,
    )
    verifier = FakeVerifierAgent(fixed_result=improved)
    judge = FakeJudgeAgent(
        results_by_id={
            "chicken-breast-raw": _low_verdict("chicken-breast-raw", score=0.2),
        },
    )
    step = _make_step(verifier=verifier, judge=judge)

    state = make_pipeline_state(
        items=[item], verifications=[original], ground_truth=[gt],
    )
    result = await step.run(state)

    assert result.verifications[0] is improved  # replaced
    assert len(verifier.calls) == 1
    assert verifier.calls[0] is item


async def test_hint_contains_judge_score_and_reasoning() -> None:
    """The hint passed to the verifier includes the judge score + reasoning."""
    item = make_food_item(item_id="salmon-raw")
    gt = make_ground_truth_entry(item_id="salmon-raw")
    original = make_item_verification(item=item)
    verifier = FakeVerifierAgent()
    judge = FakeJudgeAgent(
        results_by_id={
            "salmon-raw": JudgeVerdict(
                item_id="salmon-raw",
                score=0.25,
                correct_verdict=False,
                reasoning="Fat value not cross-checked against Foundation.",
            ),
        },
    )
    step = _make_step(verifier=verifier, judge=judge)

    state = make_pipeline_state(
        items=[item], verifications=[original], ground_truth=[gt],
    )
    await step.run(state)

    assert len(verifier.hints) == 1
    hint = verifier.hints[0]
    assert hint is not None
    assert "0.25" in hint
    assert "Fat value not cross-checked against Foundation." in hint


# ---------------------------------------------------------------------------
# Mixed items
# ---------------------------------------------------------------------------


async def test_only_low_scoring_items_are_retried() -> None:
    """In a multi-item run, only items below the threshold get re-verified."""
    item_a = make_food_item(item_id="item-a")
    item_b = make_food_item(item_id="item-b")
    gt_a = make_ground_truth_entry(item_id="item-a")
    gt_b = make_ground_truth_entry(item_id="item-b")
    v_a = make_item_verification(item=item_a)  # good score
    v_b = make_item_verification(item=item_b)  # bad score

    improved_b = make_item_verification(
        item=item_b, verdict=Verdict.MAJOR_DISCREPANCY,
    )
    verifier = FakeVerifierAgent(fixed_result=improved_b)
    judge = FakeJudgeAgent(
        results_by_id={
            "item-a": _high_verdict("item-a", score=0.85),
            "item-b": _low_verdict("item-b", score=0.1),
        },
    )
    step = _make_step(verifier=verifier, judge=judge)

    state = make_pipeline_state(
        items=[item_a, item_b],
        verifications=[v_a, v_b],
        ground_truth=[gt_a, gt_b],
    )
    result = await step.run(state)

    assert result.verifications[0] is v_a      # unchanged
    assert result.verifications[1] is improved_b  # replaced
    assert len(verifier.calls) == 1
    assert verifier.calls[0] is item_b


async def test_done_log_reports_correct_retry_count() -> None:
    """Logger 'self_verify.done' event includes correct retried + total counts."""
    item_a = make_food_item(item_id="bread")
    item_b = make_food_item(item_id="milk")
    gt_a = make_ground_truth_entry(item_id="bread")
    gt_b = make_ground_truth_entry(item_id="milk")

    logger = FakeLogger()
    judge = FakeJudgeAgent(
        results_by_id={
            "bread": _low_verdict("bread", score=0.2),
            "milk": _high_verdict("milk", score=0.8),
        },
    )
    step = SelfVerifyStep(
        verifier_agent=FakeVerifierAgent(),
        judge_agent=judge,
        logger=logger,
    )

    state = make_pipeline_state(
        items=[item_a, item_b],
        verifications=[
            make_item_verification(item=item_a),
            make_item_verification(item=item_b),
        ],
        ground_truth=[gt_a, gt_b],
    )
    await step.run(state)

    done_events = [
        kwargs for level, msg, kwargs in logger.messages
        if msg == "self_verify.done"
    ]
    assert len(done_events) == 1
    assert done_events[0]["retried"] == 1
    assert done_events[0]["total"] == 2


# ---------------------------------------------------------------------------
# Port compliance
# ---------------------------------------------------------------------------


def test_self_verify_step_implements_pipeline_step() -> None:
    """SelfVerifyStep satisfies PipelineStep ABC."""
    from snaq_verify.domain.ports.pipeline_step_port import PipelineStep

    step = _make_step()
    assert isinstance(step, PipelineStep)
