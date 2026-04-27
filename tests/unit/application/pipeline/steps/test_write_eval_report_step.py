"""Unit tests for WriteEvalReportStep."""

import json
from pathlib import Path

import pytest

from snaq_verify.application.pipeline.steps.write_eval_report_step import (
    WriteEvalReportStep,
)
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.factories import make_eval_report
from tests.fakes.fake_logger import FakeLogger


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_writes_json_file(tmp_path: Path) -> None:
    """EvalReport is written as valid JSON to eval_output_path."""
    report = make_eval_report()
    output = tmp_path / "eval.json"

    step = WriteEvalReportStep(logger=FakeLogger())
    await step.run(PipelineState(eval_report=report, eval_output_path=output))

    assert output.exists()
    data = json.loads(output.read_text())
    assert "metadata" in data
    assert "judgments" in data
    assert "aggregate_score" in data


async def test_content_round_trips(tmp_path: Path) -> None:
    """Written JSON round-trips to an identical EvalReport."""
    from snaq_verify.domain.models.eval_models import EvalReport

    report = make_eval_report()
    output = tmp_path / "eval.json"

    step = WriteEvalReportStep(logger=FakeLogger())
    await step.run(PipelineState(eval_report=report, eval_output_path=output))

    reloaded = EvalReport.model_validate_json(output.read_text())
    assert reloaded.model_dump() == report.model_dump()


async def test_creates_parent_directories(tmp_path: Path) -> None:
    """Parent directories are created if they do not exist."""
    report = make_eval_report()
    output = tmp_path / "deep" / "nested" / "eval.json"

    step = WriteEvalReportStep(logger=FakeLogger())
    await step.run(PipelineState(eval_report=report, eval_output_path=output))

    assert output.exists()


async def test_overwrites_existing_file(tmp_path: Path) -> None:
    """Existing file at eval_output_path is overwritten."""
    output = tmp_path / "eval.json"
    output.write_text("OLD", encoding="utf-8")

    report = make_eval_report()
    step = WriteEvalReportStep(logger=FakeLogger())
    await step.run(PipelineState(eval_report=report, eval_output_path=output))

    content = output.read_text()
    assert content != "OLD"
    assert "metadata" in json.loads(content)


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert WriteEvalReportStep(logger=FakeLogger()).name == "write_eval_report"


async def test_log_messages_on_success(tmp_path: Path) -> None:
    """Step emits info log when writing starts and finishes."""
    logger = FakeLogger()
    output = tmp_path / "eval.json"
    await WriteEvalReportStep(logger=logger).run(
        PipelineState(eval_report=make_eval_report(), eval_output_path=output)
    )

    info_msgs = [msg for _, msg, _ in logger.at_level("info")]
    assert any("write_eval_report.writing" in m for m in info_msgs)
    assert any("write_eval_report.done" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_raises_when_eval_report_is_none(tmp_path: Path) -> None:
    """Raises ValueError if state.eval_report is None."""
    step = WriteEvalReportStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="eval_report"):
        await step.run(PipelineState(eval_output_path=tmp_path / "eval.json"))


async def test_raises_when_eval_output_path_is_none() -> None:
    """Raises ValueError if state.eval_output_path is None."""
    step = WriteEvalReportStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="eval_output_path"):
        await step.run(PipelineState(eval_report=make_eval_report()))
