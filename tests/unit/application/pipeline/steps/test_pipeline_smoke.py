"""End-to-end smoke test: full pipeline with fakes, no real adapters."""

import json
from pathlib import Path

from snaq_verify.application.pipeline.runner import PipelineRunner
from snaq_verify.application.pipeline.steps.aggregate_step import AggregateStep
from snaq_verify.application.pipeline.steps.judge_step import JudgeStep
from snaq_verify.application.pipeline.steps.load_ground_truth_step import (
    LoadGroundTruthStep,
)
from snaq_verify.application.pipeline.steps.load_input_step import LoadInputStep
from snaq_verify.application.pipeline.steps.load_report_step import LoadReportStep
from snaq_verify.application.pipeline.steps.verify_step import VerifyStep
from snaq_verify.application.pipeline.steps.write_eval_report_step import (
    WriteEvalReportStep,
)
from snaq_verify.application.pipeline.steps.write_report_step import WriteReportStep
from snaq_verify.core.config import Settings
from snaq_verify.domain.models.pipeline_state import PipelineState
from tests.fakes.fake_judge_agent import FakeJudgeAgent
from tests.fakes.fake_logger import FakeLogger
from tests.fakes.fake_verifier_agent import FakeVerifierAgent


def _settings() -> Settings:
    return Settings(
        USDA_API_KEY="test",
        OPENAI_API_KEY="test",
        TAVILY_API_KEY="test",
        OPENAI_MODEL="gpt-smoke-test",
    )


def _make_input(tmp_path: Path, n: int = 3) -> Path:
    items = []
    for i in range(n):
        items.append({
            "id": f"item-{i}",
            "name": f"Food {i}",
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
        })
    p = tmp_path / "input.json"
    p.write_text(json.dumps(items), encoding="utf-8")
    return p


def _make_ground_truth(tmp_path: Path, item_ids: list[str]) -> Path:
    entries = [
        {
            "item_id": item_id,
            "item_name": f"Food {item_id}",
            "source": "USDA Foundation",
            "source_url": "https://fdc.nal.usda.gov/test",
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
        for item_id in item_ids
    ]
    p = tmp_path / "ground_truth.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Verification pipeline smoke test
# ---------------------------------------------------------------------------


async def test_verification_pipeline_end_to_end(tmp_path: Path) -> None:
    """Full verification pipeline with fakes writes a valid report.json."""
    n = 3
    input_path = _make_input(tmp_path, n=n)
    output_path = tmp_path / "report.json"

    logger = FakeLogger()
    settings = _settings()

    steps = [
        LoadInputStep(logger=logger),
        VerifyStep(verifier_agent=FakeVerifierAgent(), logger=logger),
        AggregateStep(logger=logger, settings=settings),
        WriteReportStep(logger=logger),
    ]

    runner = PipelineRunner(logger=logger)
    state = PipelineState(input_path=input_path, output_path=output_path)
    final = await runner.run(state, steps)

    # State checks
    assert len(final.items) == n
    assert len(final.verifications) == n
    assert final.report is not None
    assert final.report.metadata.input_count == n

    # File checks
    assert output_path.exists()
    from snaq_verify.domain.models.verification_report import VerificationReport
    loaded = VerificationReport.model_validate_json(output_path.read_text())
    assert len(loaded.items) == n
    assert loaded.metadata.model == "gpt-smoke-test"


# ---------------------------------------------------------------------------
# Eval pipeline smoke test
# ---------------------------------------------------------------------------


async def test_eval_pipeline_end_to_end(tmp_path: Path) -> None:
    """Full eval pipeline with fakes writes a valid eval_report.json."""
    n = 3
    input_path = _make_input(tmp_path, n=n)
    output_path = tmp_path / "report.json"
    eval_output_path = tmp_path / "eval_report.json"
    gt_path = _make_ground_truth(tmp_path, [f"item-{i}" for i in range(n)])

    logger = FakeLogger()
    settings = _settings()

    # Step 1: run verification pipeline
    verify_steps = [
        LoadInputStep(logger=logger),
        VerifyStep(verifier_agent=FakeVerifierAgent(), logger=logger),
        AggregateStep(logger=logger, settings=settings),
        WriteReportStep(logger=logger),
    ]
    runner = PipelineRunner(logger=logger)
    state = PipelineState(input_path=input_path, output_path=output_path)
    state = await runner.run(state, verify_steps)

    # Step 2: run eval pipeline (report already in memory)
    eval_steps = [
        LoadReportStep(logger=logger),
        LoadGroundTruthStep(logger=logger),
        JudgeStep(judge_agent=FakeJudgeAgent(), logger=logger, settings=settings),
        WriteEvalReportStep(logger=logger),
    ]
    state.eval_output_path = eval_output_path
    state.ground_truth_path = gt_path
    state = await runner.run(state, eval_steps)

    assert state.eval_report is not None
    assert state.eval_report.total == n
    assert state.eval_report.aggregate_score == 1.0  # FakeJudgeAgent default

    # File checks
    assert eval_output_path.exists()
    from snaq_verify.domain.models.eval_models import EvalReport
    loaded = EvalReport.model_validate_json(eval_output_path.read_text())
    assert loaded.total == n


# ---------------------------------------------------------------------------
# run-and-eval (single process, report in memory) smoke test
# ---------------------------------------------------------------------------


async def test_run_and_eval_in_memory(tmp_path: Path) -> None:
    """run-and-eval flow: report stays in memory for eval, no second file read."""
    n = 2
    input_path = _make_input(tmp_path, n=n)
    output_path = tmp_path / "report.json"
    eval_output_path = tmp_path / "eval_report.json"
    gt_path = _make_ground_truth(tmp_path, [f"item-{i}" for i in range(n)])

    logger = FakeLogger()
    settings = _settings()
    runner = PipelineRunner(logger=logger)

    # Verification pipeline
    state = PipelineState(
        input_path=input_path,
        output_path=output_path,
        eval_output_path=eval_output_path,
        ground_truth_path=gt_path,
    )
    state = await runner.run(state, [
        LoadInputStep(logger=logger),
        VerifyStep(verifier_agent=FakeVerifierAgent(), logger=logger),
        AggregateStep(logger=logger, settings=settings),
        WriteReportStep(logger=logger),
    ])

    # Eval pipeline — LoadReportStep hits in-memory shortcut
    state = await runner.run(state, [
        LoadReportStep(logger=logger),
        LoadGroundTruthStep(logger=logger),
        JudgeStep(judge_agent=FakeJudgeAgent(), logger=logger, settings=settings),
        WriteEvalReportStep(logger=logger),
    ])

    assert state.eval_report is not None
    assert state.eval_report.total == n
    assert eval_output_path.exists()

    # LoadReportStep used the in-memory shortcut — log should say so
    assert any(
        "load_report.using_in_memory_report" in msg
        for _, msg, _ in logger.messages
    )
