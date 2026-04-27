"""Unit tests for LoadInputStep."""

import json
from pathlib import Path

import pytest

from snaq_verify.application.pipeline.steps.load_input_step import LoadInputStep
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.fake_logger import FakeLogger
from tests.fakes.factories import make_food_item, make_nutrition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_items(path: Path, items: list[dict]) -> None:  # type: ignore[type-arg]
    path.write_text(json.dumps(items), encoding="utf-8")


def _food_item_dict(**overrides: object) -> dict:  # type: ignore[type-arg]
    base: dict = {  # type: ignore[type-arg]
        "id": "test-item",
        "name": "Test Food",
        "brand": None,
        "category": None,
        "barcode": None,
        "default_portion": {
            "amount": 100.0,
            "unit": "g",
            "description": "1 serving",
        },
        "nutrition_per_100g": {
            "calories_kcal": 100.0,
            "protein_g": 10.0,
            "fat_g": 5.0,
            "saturated_fat_g": 1.0,
            "carbohydrates_g": 10.0,
            "sugar_g": 2.0,
            "fiber_g": 1.0,
            "sodium_mg": 50.0,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_load_single_item(tmp_path: Path) -> None:
    """A valid single-item JSON file is loaded and validated."""
    f = tmp_path / "items.json"
    _write_items(f, [_food_item_dict()])

    step = LoadInputStep(logger=FakeLogger())
    state = PipelineState(input_path=f)
    result = await step.run(state)

    assert len(result.items) == 1
    assert result.items[0].id == "test-item"
    assert result.items[0].name == "Test Food"


async def test_load_multiple_items_preserves_order(tmp_path: Path) -> None:
    """Items are returned in the same order as in the file."""
    raw = [_food_item_dict(id=f"item-{i}", name=f"Food {i}") for i in range(5)]
    f = tmp_path / "items.json"
    _write_items(f, raw)

    step = LoadInputStep(logger=FakeLogger())
    state = PipelineState(input_path=f)
    result = await step.run(state)

    assert [item.id for item in result.items] == [f"item-{i}" for i in range(5)]


async def test_load_empty_array(tmp_path: Path) -> None:
    """An empty JSON array yields state.items == []."""
    f = tmp_path / "items.json"
    _write_items(f, [])

    step = LoadInputStep(logger=FakeLogger())
    state = PipelineState(input_path=f)
    result = await step.run(state)

    assert result.items == []


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert LoadInputStep(logger=FakeLogger()).name == "load_input"


async def test_log_messages_on_success(tmp_path: Path) -> None:
    """Step emits info log on read and on completion."""
    f = tmp_path / "items.json"
    _write_items(f, [_food_item_dict()])

    logger = FakeLogger()
    step = LoadInputStep(logger=logger)
    await step.run(PipelineState(input_path=f))

    info_msgs = [msg for _, msg, _ in logger.at_level("info")]
    assert any("load_input.reading" in m for m in info_msgs)
    assert any("load_input.done" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_raises_when_input_path_is_none() -> None:
    """Raises ValueError if state.input_path is None."""
    step = LoadInputStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="input_path"):
        await step.run(PipelineState())


async def test_raises_for_missing_file(tmp_path: Path) -> None:
    """Raises FileNotFoundError if the file does not exist."""
    step = LoadInputStep(logger=FakeLogger())
    with pytest.raises(FileNotFoundError):
        await step.run(PipelineState(input_path=tmp_path / "missing.json"))


async def test_raises_for_invalid_json(tmp_path: Path) -> None:
    """Raises ValueError on malformed JSON."""
    f = tmp_path / "bad.json"
    f.write_text("{not: valid}", encoding="utf-8")

    step = LoadInputStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="Invalid JSON"):
        await step.run(PipelineState(input_path=f))


async def test_raises_when_root_is_not_array(tmp_path: Path) -> None:
    """Raises ValueError if the JSON root is not an array."""
    f = tmp_path / "obj.json"
    f.write_text('{"key": "value"}', encoding="utf-8")

    step = LoadInputStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="JSON array"):
        await step.run(PipelineState(input_path=f))


async def test_raises_for_schema_violation(tmp_path: Path) -> None:
    """Raises ValidationError when an item fails the FoodItem schema."""
    from pydantic import ValidationError

    f = tmp_path / "bad_schema.json"
    f.write_text('[{"id": "x", "missing_required": true}]', encoding="utf-8")

    step = LoadInputStep(logger=FakeLogger())
    with pytest.raises(ValidationError):
        await step.run(PipelineState(input_path=f))
