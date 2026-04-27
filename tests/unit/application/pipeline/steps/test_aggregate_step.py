"""Unit tests for AggregateStep."""

from snaq_verify.application.pipeline.steps.aggregate_step import AggregateStep
from snaq_verify.core.config import Settings
from snaq_verify.domain.models.enums import Verdict
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.factories import make_food_item, make_item_verification
from tests.fakes.fake_logger import FakeLogger


def _settings() -> Settings:
    return Settings(
        USDA_API_KEY="test",
        OPENAI_API_KEY="test",
        TAVILY_API_KEY="test",
        OPENAI_MODEL="gpt-test",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_report_has_correct_item_count() -> None:
    """metadata.input_count reflects state.items."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(3)]
    verifications = [make_item_verification(item=i) for i in items]

    step = AggregateStep(logger=FakeLogger(), settings=_settings())
    state = PipelineState(items=items, verifications=verifications)
    result = await step.run(state)

    assert result.report is not None
    assert result.report.metadata.input_count == 3


async def test_report_items_match_verifications() -> None:
    """report.items is exactly state.verifications."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(2)]
    verifications = [make_item_verification(item=i) for i in items]

    step = AggregateStep(logger=FakeLogger(), settings=_settings())
    state = PipelineState(items=items, verifications=verifications)
    result = await step.run(state)

    assert result.report is not None
    assert result.report.items == verifications


async def test_flag_count_counts_discrepancies() -> None:
    """flag_count = minor + major discrepancies, not matches."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(4)]
    verifications = [
        make_item_verification(item=items[0], verdict=Verdict.MATCH),
        make_item_verification(item=items[1], verdict=Verdict.MINOR_DISCREPANCY),
        make_item_verification(item=items[2], verdict=Verdict.MAJOR_DISCREPANCY),
        make_item_verification(item=items[3], verdict=Verdict.MINOR_DISCREPANCY),
    ]

    step = AggregateStep(logger=FakeLogger(), settings=_settings())
    result = await step.run(PipelineState(items=items, verifications=verifications))

    assert result.report is not None
    assert result.report.metadata.flag_count == 3


async def test_flag_count_zero_when_all_match() -> None:
    """flag_count is 0 if every item is a match."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(3)]
    verifications = [make_item_verification(item=i, verdict=Verdict.MATCH) for i in items]

    step = AggregateStep(logger=FakeLogger(), settings=_settings())
    result = await step.run(PipelineState(items=items, verifications=verifications))

    assert result.report is not None
    assert result.report.metadata.flag_count == 0


async def test_model_pin_comes_from_settings() -> None:
    """metadata.model reflects settings.OPENAI_MODEL."""
    item = make_food_item()
    step = AggregateStep(logger=FakeLogger(), settings=_settings())
    result = await step.run(
        PipelineState(items=[item], verifications=[make_item_verification(item=item)])
    )

    assert result.report is not None
    assert result.report.metadata.model == "gpt-test"


async def test_empty_verifications_produces_empty_report() -> None:
    """No verifications → empty report items, flag_count 0."""
    step = AggregateStep(logger=FakeLogger(), settings=_settings())
    result = await step.run(PipelineState(items=[], verifications=[]))

    assert result.report is not None
    assert result.report.items == []
    assert result.report.metadata.flag_count == 0


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert AggregateStep(logger=FakeLogger(), settings=_settings()).name == "aggregate"


async def test_timestamp_is_set() -> None:
    """metadata.timestamp is a non-None datetime."""
    item = make_food_item()
    step = AggregateStep(logger=FakeLogger(), settings=_settings())
    result = await step.run(
        PipelineState(items=[item], verifications=[make_item_verification(item=item)])
    )

    assert result.report is not None
    assert result.report.metadata.timestamp is not None
