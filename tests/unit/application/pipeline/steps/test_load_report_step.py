"""Unit tests for LoadReportStep."""

import json
from pathlib import Path

import pytest

from snaq_verify.application.pipeline.steps.load_report_step import LoadReportStep
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.factories import make_food_item, make_item_verification, make_verification_report
from tests.fakes.fake_logger import FakeLogger


# ---------------------------------------------------------------------------
# Happy path — file on disk
# ---------------------------------------------------------------------------


async def test_loads_report_from_disk(tmp_path: Path) -> None:
    """Reads report JSON from output_path when state.report is None."""
    report = make_verification_report()
    rpath = tmp_path / "report.json"
    rpath.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    step = LoadReportStep(logger=FakeLogger())
    state = PipelineState(output_path=rpath)
    result = await step.run(state)

    assert result.report is not None
    assert result.report.metadata.input_count == report.metadata.input_count


async def test_populates_verifications_from_disk(tmp_path: Path) -> None:
    """state.verifications is set from loaded report.items."""
    items = [make_food_item(item_id=f"item-{i}") for i in range(2)]
    report = make_verification_report(items=[make_item_verification(item=i) for i in items])
    rpath = tmp_path / "report.json"
    rpath.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    step = LoadReportStep(logger=FakeLogger())
    result = await step.run(PipelineState(output_path=rpath))

    assert len(result.verifications) == 2


# ---------------------------------------------------------------------------
# run-and-eval shortcut: report already in memory
# ---------------------------------------------------------------------------


async def test_uses_in_memory_report_when_set() -> None:
    """Skips disk read when state.report is already populated."""
    item = make_food_item()
    report = make_verification_report(items=[make_item_verification(item=item)])

    step = LoadReportStep(logger=FakeLogger())
    result = await step.run(PipelineState(report=report))

    # report is unchanged; verifications are populated from it
    assert result.report is report
    assert len(result.verifications) == 1


async def test_in_memory_path_skips_disk_entirely(tmp_path: Path) -> None:
    """When state.report is set, output_path is not read even if present."""
    item = make_food_item()
    report = make_verification_report(items=[make_item_verification(item=item)])

    # Point output_path at a non-existent file — should NOT raise
    nonexistent = tmp_path / "ghost.json"
    step = LoadReportStep(logger=FakeLogger())
    result = await step.run(PipelineState(report=report, output_path=nonexistent))

    assert result.report is report


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_raises_when_no_report_and_no_path() -> None:
    """Raises ValueError if both state.report and state.output_path are None."""
    step = LoadReportStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="output_path"):
        await step.run(PipelineState())


async def test_raises_for_missing_file(tmp_path: Path) -> None:
    """Raises FileNotFoundError if the report file does not exist."""
    step = LoadReportStep(logger=FakeLogger())
    with pytest.raises(FileNotFoundError):
        await step.run(PipelineState(output_path=tmp_path / "missing.json"))


async def test_raises_for_invalid_json(tmp_path: Path) -> None:
    """Raises ValueError on malformed JSON."""
    bad = tmp_path / "bad.json"
    bad.write_text("{bad json}", encoding="utf-8")

    step = LoadReportStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="Invalid JSON"):
        await step.run(PipelineState(output_path=bad))


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert LoadReportStep(logger=FakeLogger()).name == "load_report"
