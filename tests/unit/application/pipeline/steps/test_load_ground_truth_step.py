"""Unit tests for LoadGroundTruthStep."""

import json
from pathlib import Path

import pytest

from snaq_verify.application.pipeline.steps.load_ground_truth_step import (
    LoadGroundTruthStep,
)
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.factories import make_ground_truth_entry
from tests.fakes.fake_logger import FakeLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ground_truth(path: Path, entries: list[dict]) -> None:  # type: ignore[type-arg]
    path.write_text(json.dumps(entries), encoding="utf-8")


def _gt_dict(item_id: str = "test-item") -> dict:  # type: ignore[type-arg]
    return {
        "item_id": item_id,
        "item_name": "Test Food",
        "source": "USDA Foundation",
        "source_url": "https://fdc.nal.usda.gov/food-details/test/nutrients",
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
        "notes": None,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_loads_single_entry(tmp_path: Path) -> None:
    """Single ground-truth entry is loaded and validated."""
    f = tmp_path / "gt.json"
    _write_ground_truth(f, [_gt_dict()])

    step = LoadGroundTruthStep(logger=FakeLogger())
    result = await step.run(PipelineState(ground_truth_path=f))

    assert len(result.ground_truth) == 1
    assert result.ground_truth[0].item_id == "test-item"


async def test_loads_multiple_entries_preserves_order(tmp_path: Path) -> None:
    """Multiple entries are loaded in file order."""
    raw = [_gt_dict(item_id=f"item-{i}") for i in range(4)]
    f = tmp_path / "gt.json"
    _write_ground_truth(f, raw)

    step = LoadGroundTruthStep(logger=FakeLogger())
    result = await step.run(PipelineState(ground_truth_path=f))

    assert [e.item_id for e in result.ground_truth] == [f"item-{i}" for i in range(4)]


async def test_loads_empty_array(tmp_path: Path) -> None:
    """Empty ground-truth file yields state.ground_truth == []."""
    f = tmp_path / "gt.json"
    _write_ground_truth(f, [])

    step = LoadGroundTruthStep(logger=FakeLogger())
    result = await step.run(PipelineState(ground_truth_path=f))

    assert result.ground_truth == []


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert LoadGroundTruthStep(logger=FakeLogger()).name == "load_ground_truth"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_raises_when_path_is_none() -> None:
    """Raises ValueError if state.ground_truth_path is None."""
    step = LoadGroundTruthStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="ground_truth_path"):
        await step.run(PipelineState())


async def test_raises_for_missing_file(tmp_path: Path) -> None:
    """Raises FileNotFoundError when file does not exist."""
    step = LoadGroundTruthStep(logger=FakeLogger())
    with pytest.raises(FileNotFoundError):
        await step.run(PipelineState(ground_truth_path=tmp_path / "ghost.json"))


async def test_raises_for_invalid_json(tmp_path: Path) -> None:
    """Raises ValueError on malformed JSON."""
    bad = tmp_path / "bad.json"
    bad.write_text("{bad json}", encoding="utf-8")

    step = LoadGroundTruthStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="Invalid JSON"):
        await step.run(PipelineState(ground_truth_path=bad))


async def test_raises_when_root_is_not_array(tmp_path: Path) -> None:
    """Raises ValueError if the JSON root is not an array."""
    f = tmp_path / "obj.json"
    f.write_text('{"item_id": "x"}', encoding="utf-8")

    step = LoadGroundTruthStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="JSON array"):
        await step.run(PipelineState(ground_truth_path=f))
