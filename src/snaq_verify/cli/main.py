"""snaq-verify CLI — Typer app with run / eval / run-and-eval subcommands."""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from snaq_verify.application.pipeline.steps.aggregate_step import AggregateStep
from snaq_verify.application.pipeline.steps.judge_step import JudgeStep
from snaq_verify.application.pipeline.steps.load_ground_truth_step import (
    LoadGroundTruthStep,
)
from snaq_verify.application.pipeline.steps.load_input_step import LoadInputStep
from snaq_verify.application.pipeline.steps.load_report_step import LoadReportStep
from snaq_verify.application.pipeline.steps.self_verify_step import SelfVerifyStep
from snaq_verify.application.pipeline.steps.verify_step import VerifyStep
from snaq_verify.application.pipeline.steps.write_eval_report_step import (
    WriteEvalReportStep,
)
from snaq_verify.application.pipeline.steps.write_report_step import WriteReportStep
from snaq_verify.bootstrap import Bootstrap
from snaq_verify.domain.models.pipeline_state import PipelineState

app = typer.Typer(
    name="snaq-verify",
    help="SNAQ deterministic nutrition verification system.",
    add_completion=False,
)

_console = Console()
_err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    input: Path = typer.Option(  # noqa: A002
        ...,
        "--input",
        "-i",
        help="Path to food_items.json",
        exists=False,
        file_okay=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write verification_report.json",
    ),
) -> None:
    """Verify food items from INPUT and write the report to OUTPUT."""
    try:
        asyncio.run(_run_verification(input_path=input, output_path=output))
    except Exception as exc:
        _err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


async def _run_verification(input_path: Path, output_path: Path) -> None:
    container = Bootstrap.build()
    logger = container.logger

    # Phase 1: load items (fast, no API)
    state = PipelineState(input_path=input_path, output_path=output_path)
    load_step = LoadInputStep(logger=logger)
    state = await load_step.run(state)

    total = len(state.items)
    _console.print(
        f"[bold green]✓[/bold green] Loaded [bold]{total}[/bold] items from {input_path}"
    )

    # Phase 2: verify + aggregate + write with per-item progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Verifying items...", total=total)

        def _on_item_done(item_id: str) -> None:
            progress.advance(task_id)
            progress.update(task_id, description=f"Verified {item_id}")

        remaining_steps = [
            VerifyStep(
                verifier_agent=container.verifier_agent,
                logger=logger,
                on_item_complete=_on_item_done,
            ),
            AggregateStep(logger=logger, settings=container.settings),
            WriteReportStep(logger=logger),
        ]
        state = await container.runner.run(state, remaining_steps)

    assert state.report is not None
    flagged = state.report.metadata.flag_count
    _console.print(
        f"[bold green]✓[/bold green] Report written → {output_path}  "
        f"[yellow]({flagged} flagged)[/yellow]"
    )


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@app.command()
def eval(  # noqa: A001
    report: Path = typer.Option(
        ...,
        "--report",
        "-r",
        help="Path to an existing verification_report.json",
    ),
    ground_truth: Path = typer.Option(
        ...,
        "--ground-truth",
        "-g",
        help="Path to ground_truth.json",
    ),
    output: Path = typer.Option(  # noqa: A002
        ...,
        "--output",
        "-o",
        help="Path to write eval_report.json",
    ),
) -> None:
    """Score an existing verification report against a golden set."""
    try:
        asyncio.run(
            _run_eval(
                report_path=report,
                ground_truth_path=ground_truth,
                eval_output_path=output,
            )
        )
    except Exception as exc:
        _err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


async def _run_eval(
    report_path: Path,
    ground_truth_path: Path,
    eval_output_path: Path,
) -> None:
    container = Bootstrap.build()
    logger = container.logger

    state = PipelineState(
        output_path=report_path,
        ground_truth_path=ground_truth_path,
        eval_output_path=eval_output_path,
    )

    eval_steps = [
        LoadReportStep(logger=logger),
        LoadGroundTruthStep(logger=logger),
        JudgeStep(judge_agent=container.judge_agent, logger=logger, settings=container.settings),
        WriteEvalReportStep(logger=logger),
    ]

    with _console.status("[bold]Running eval pipeline…[/bold]"):
        state = await container.runner.run(state, eval_steps)

    assert state.eval_report is not None
    agg = state.eval_report.aggregate_score
    correct = state.eval_report.correct_verdicts
    total = state.eval_report.total
    _console.print(
        f"[bold green]✓[/bold green] Eval written → {eval_output_path}  "
        f"[yellow](score={agg:.2f}, {correct}/{total} correct)[/yellow]"
    )


# ---------------------------------------------------------------------------
# run-and-eval
# ---------------------------------------------------------------------------


@app.command(name="run-and-eval")
def run_and_eval(
    input: Path = typer.Option(  # noqa: A002
        ...,
        "--input",
        "-i",
        help="Path to food_items.json",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write verification_report.json",
    ),
    eval_output: Path = typer.Option(
        ...,
        "--eval-output",
        "-e",
        help="Path to write eval_report.json",
    ),
    ground_truth: Path = typer.Option(
        ...,
        "--ground-truth",
        "-g",
        help="Path to ground_truth.json",
    ),
) -> None:
    """Verify food items and immediately judge against the golden set."""
    try:
        asyncio.run(
            _run_and_eval(
                input_path=input,
                output_path=output,
                eval_output_path=eval_output,
                ground_truth_path=ground_truth,
            )
        )
    except Exception as exc:
        _err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


async def _run_and_eval(
    input_path: Path,
    output_path: Path,
    eval_output_path: Path,
    ground_truth_path: Path,
) -> None:
    container = Bootstrap.build()
    logger = container.logger

    # ----- verification pipeline ----------------------------------------
    state = PipelineState(
        input_path=input_path,
        output_path=output_path,
        eval_output_path=eval_output_path,
        ground_truth_path=ground_truth_path,
    )

    load_step = LoadInputStep(logger=logger)
    state = await load_step.run(state)

    total = len(state.items)
    _console.print(
        f"[bold green]✓[/bold green] Loaded [bold]{total}[/bold] items from {input_path}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Verifying items...", total=total)

        def _on_item_done(item_id: str) -> None:
            progress.advance(task_id)
            progress.update(task_id, description=f"Verified {item_id}")

        state = await container.runner.run(state, [
            LoadGroundTruthStep(logger=logger),          # moved here so SelfVerifyStep has ground truth
            VerifyStep(
                verifier_agent=container.verifier_agent,
                logger=logger,
                on_item_complete=_on_item_done,
            ),
            SelfVerifyStep(                              # re-verify low-scoring items inline
                verifier_agent=container.verifier_agent,
                judge_agent=container.judge_agent,
                logger=logger,
            ),
            AggregateStep(logger=logger, settings=container.settings),
            WriteReportStep(logger=logger),
        ])

    assert state.report is not None
    flagged = state.report.metadata.flag_count
    _console.print(
        f"[bold green]✓[/bold green] Verification report → {output_path}  "
        f"[yellow]({flagged} flagged)[/yellow]"
    )
    _console.print("[bold green]✓[/bold green] Self-verification complete")

    # ----- eval pipeline (report already in memory) ---------------------
    with _console.status("[bold]Running eval pipeline…[/bold]"):
        state = await container.runner.run(state, [
            LoadReportStep(logger=logger),       # hits in-memory shortcut
            # LoadGroundTruthStep already ran above — idempotency guard skips it
            JudgeStep(
                judge_agent=container.judge_agent,
                logger=logger,
                settings=container.settings,
            ),
            WriteEvalReportStep(logger=logger),
        ])

    assert state.eval_report is not None
    agg = state.eval_report.aggregate_score
    correct = state.eval_report.correct_verdicts
    eval_total = state.eval_report.total
    _console.print(
        f"[bold green]✓[/bold green] Eval report → {eval_output_path}  "
        f"[yellow](score={agg:.2f}, {correct}/{eval_total} correct)[/yellow]"
    )


# ---------------------------------------------------------------------------
# diff-runs
# ---------------------------------------------------------------------------


@app.command(name="diff-runs")
def diff_runs(
    baseline: Path = typer.Argument(
        ...,
        help="Path to the baseline eval_report.json (e.g. from plain run + eval).",
        exists=False,
    ),
    corrected: Path = typer.Argument(
        ...,
        help="Path to the corrected eval_report.json (e.g. from run-and-eval with SelfVerifyStep).",
        exists=False,
    ),
) -> None:
    """Compare two eval_report.json files and surface per-item score deltas.

    Exits 0 when the corrected run is the same or better than baseline.
    Exits 1 when the corrected run is worse (regression guard).
    """
    from snaq_verify.cli.diff_runs import compare_eval_reports

    code = compare_eval_reports(baseline, corrected, console=_console, err_console=_err_console)
    raise typer.Exit(code=code)
