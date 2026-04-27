"""Unit tests for JudgeStep."""

import pytest

from snaq_verify.application.pipeline.steps.judge_step import JudgeStep
from snaq_verify.core.config import Settings
from snaq_verify.domain.models.eval_models import JudgeVerdict
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.factories import (
    make_food_item,
    make_ground_truth_entry,
    make_item_verification,
    make_verification_report,
)
from tests.fakes.fake_judge_agent import FakeJudgeAgent
from tests.fakes.fake_logger import FakeLogger


def _settings() -> Settings:
    return Settings(
        USDA_API_KEY="test",
        OPENAI_API_KEY="test",
        TAVILY_API_KEY="test",
        OPENAI_MODEL="gpt-judge-test",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_judges_matching_item() -> None:
    """Judge is called for an item that has a ground-truth entry."""
    item = make_food_item(item_id="chicken")
    report = make_verification_report(items=[make_item_verification(item=item)])
    gt = [make_ground_truth_entry(item_id="chicken")]

    agent = FakeJudgeAgent()
    step = JudgeStep(judge_agent=agent, logger=FakeLogger(), settings=_settings())
    result = await step.run(PipelineState(report=report, ground_truth=gt))

    assert result.eval_report is not None
    assert len(result.eval_report.judgments) == 1
    assert result.eval_report.judgments[0].item_id == "chicken"
    assert len(agent.calls) == 1


async def test_skips_items_without_ground_truth() -> None:
    """Items without a matching ground-truth entry are silently skipped."""
    items = [make_food_item(item_id="known"), make_food_item(item_id="unknown")]
    report = make_verification_report(
        items=[make_item_verification(item=i) for i in items]
    )
    gt = [make_ground_truth_entry(item_id="known")]

    step = JudgeStep(
        judge_agent=FakeJudgeAgent(), logger=FakeLogger(), settings=_settings()
    )
    result = await step.run(PipelineState(report=report, ground_truth=gt))

    assert result.eval_report is not None
    assert len(result.eval_report.judgments) == 1
    assert result.eval_report.judgments[0].item_id == "known"


async def test_aggregate_score_is_mean_of_individual_scores() -> None:
    """aggregate_score = mean of all individual scores."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(3)]
    report = make_verification_report(items=[make_item_verification(item=i) for i in items])
    gt = [make_ground_truth_entry(item_id=f"item-{i}") for i in range(3)]

    verdicts = {
        "item-0": JudgeVerdict(item_id="item-0", score=0.5, correct_verdict=False, reasoning="x"),
        "item-1": JudgeVerdict(item_id="item-1", score=1.0, correct_verdict=True, reasoning="y"),
        "item-2": JudgeVerdict(item_id="item-2", score=0.75, correct_verdict=True, reasoning="z"),
    }
    agent = FakeJudgeAgent(results_by_id=verdicts)

    step = JudgeStep(judge_agent=agent, logger=FakeLogger(), settings=_settings())
    result = await step.run(PipelineState(report=report, ground_truth=gt))

    assert result.eval_report is not None
    assert abs(result.eval_report.aggregate_score - (0.5 + 1.0 + 0.75) / 3) < 1e-9


async def test_correct_verdicts_count() -> None:
    """correct_verdicts counts items where correct_verdict=True."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(3)]
    report = make_verification_report(items=[make_item_verification(item=i) for i in items])
    gt = [make_ground_truth_entry(item_id=f"item-{i}") for i in range(3)]

    verdicts = {
        "item-0": JudgeVerdict(item_id="item-0", score=0.0, correct_verdict=False, reasoning="x"),
        "item-1": JudgeVerdict(item_id="item-1", score=1.0, correct_verdict=True, reasoning="y"),
        "item-2": JudgeVerdict(item_id="item-2", score=1.0, correct_verdict=True, reasoning="z"),
    }
    agent = FakeJudgeAgent(results_by_id=verdicts)

    step = JudgeStep(judge_agent=agent, logger=FakeLogger(), settings=_settings())
    result = await step.run(PipelineState(report=report, ground_truth=gt))

    assert result.eval_report is not None
    assert result.eval_report.correct_verdicts == 2
    assert result.eval_report.total == 3


async def test_empty_ground_truth_yields_zero_judgments() -> None:
    """No ground-truth entries → eval_report has 0 judgments."""
    item = make_food_item()
    report = make_verification_report(items=[make_item_verification(item=item)])

    step = JudgeStep(
        judge_agent=FakeJudgeAgent(), logger=FakeLogger(), settings=_settings()
    )
    result = await step.run(PipelineState(report=report, ground_truth=[]))

    assert result.eval_report is not None
    assert result.eval_report.total == 0
    assert result.eval_report.aggregate_score == 0.0


async def test_model_pin_comes_from_settings() -> None:
    """eval_report.metadata.model reflects settings.OPENAI_MODEL."""
    item = make_food_item()
    report = make_verification_report(items=[make_item_verification(item=item)])
    gt = [make_ground_truth_entry(item_id=item.id)]

    step = JudgeStep(
        judge_agent=FakeJudgeAgent(), logger=FakeLogger(), settings=_settings()
    )
    result = await step.run(PipelineState(report=report, ground_truth=gt))

    assert result.eval_report is not None
    assert result.eval_report.metadata.model == "gpt-judge-test"


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert (
        JudgeStep(
            judge_agent=FakeJudgeAgent(), logger=FakeLogger(), settings=_settings()
        ).name
        == "judge"
    )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_raises_when_report_is_none() -> None:
    """Raises ValueError if state.report is None."""
    step = JudgeStep(
        judge_agent=FakeJudgeAgent(), logger=FakeLogger(), settings=_settings()
    )
    with pytest.raises(ValueError, match="report"):
        await step.run(PipelineState())
