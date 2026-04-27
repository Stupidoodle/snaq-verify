"""Unit tests for WriteReportStep."""

import json
from pathlib import Path

import pytest

from snaq_verify.application.pipeline.steps.write_report_step import WriteReportStep
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.factories import make_verification_report
from tests.fakes.fake_logger import FakeLogger


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_writes_json_file(tmp_path: Path) -> None:
    """Report is written as valid JSON to output_path."""
    report = make_verification_report()
    output = tmp_path / "report.json"

    step = WriteReportStep(logger=FakeLogger())
    await step.run(PipelineState(report=report, output_path=output))

    assert output.exists()
    data = json.loads(output.read_text())
    assert "metadata" in data
    assert "items" in data


async def test_content_matches_model_dump(tmp_path: Path) -> None:
    """Written JSON round-trips back to an identical VerificationReport."""
    from snaq_verify.domain.models.verification_report import VerificationReport

    report = make_verification_report()
    output = tmp_path / "report.json"

    step = WriteReportStep(logger=FakeLogger())
    await step.run(PipelineState(report=report, output_path=output))

    reloaded = VerificationReport.model_validate_json(output.read_text())
    assert reloaded.model_dump() == report.model_dump()


async def test_creates_parent_directories(tmp_path: Path) -> None:
    """Parent directories are created if they do not exist."""
    report = make_verification_report()
    output = tmp_path / "deep" / "nested" / "report.json"

    step = WriteReportStep(logger=FakeLogger())
    await step.run(PipelineState(report=report, output_path=output))

    assert output.exists()


async def test_overwrites_existing_file(tmp_path: Path) -> None:
    """Existing file at output_path is overwritten."""
    output = tmp_path / "report.json"
    output.write_text("OLD", encoding="utf-8")

    report = make_verification_report()
    step = WriteReportStep(logger=FakeLogger())
    await step.run(PipelineState(report=report, output_path=output))

    content = output.read_text()
    assert content != "OLD"
    assert "metadata" in json.loads(content)


async def test_step_name() -> None:
    """Step reports the correct stable name."""
    assert WriteReportStep(logger=FakeLogger()).name == "write_report"


async def test_log_messages_on_success(tmp_path: Path) -> None:
    """Step emits info log when writing starts and finishes."""
    logger = FakeLogger()
    output = tmp_path / "report.json"
    await WriteReportStep(logger=logger).run(
        PipelineState(report=make_verification_report(), output_path=output)
    )

    info_msgs = [msg for _, msg, _ in logger.at_level("info")]
    assert any("write_report.writing" in m for m in info_msgs)
    assert any("write_report.done" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_raises_when_report_is_none(tmp_path: Path) -> None:
    """Raises ValueError if state.report is None."""
    step = WriteReportStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="report"):
        await step.run(PipelineState(output_path=tmp_path / "out.json"))


async def test_raises_when_output_path_is_none() -> None:
    """Raises ValueError if state.output_path is None."""
    step = WriteReportStep(logger=FakeLogger())
    with pytest.raises(ValueError, match="output_path"):
        await step.run(PipelineState(report=make_verification_report()))
