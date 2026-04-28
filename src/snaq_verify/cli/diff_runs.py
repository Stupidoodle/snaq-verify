"""diff-runs — compare two eval_report.json files (baseline vs corrected).

Intended use-case: prove that ``run-and-eval`` (with ``SelfVerifyStep``) improves
results over a plain ``run`` + ``eval`` baseline.

Usage (via CLI)::

    snaq-verify diff-runs eval_baseline.json eval_corrected.json

Usage (standalone)::

    python -m snaq_verify.cli.diff_runs eval_baseline.json eval_corrected.json

Exit codes:
    0  — corrected run is the same or better than baseline
    1  — corrected run is *worse* than baseline (regression guard)
    2  — bad arguments or unreadable files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

# Score delta above which an item is highlighted as "notable improvement/regression".
_DELTA_HIGHLIGHT_THRESHOLD = 0.1


def _load_eval_report(path: Path, err_console: Console) -> dict[str, Any] | None:
    """Parse an eval_report.json; return None and print error on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        err_console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        err_console.print(f"[bold red]Error:[/bold red] Invalid JSON in {path}: {exc}")
        return None
    if not isinstance(data, dict) or "judgments" not in data:
        err_console.print(
            f"[bold red]Error:[/bold red] {path} does not look like an eval_report.json "
            "(missing 'judgments' key)"
        )
        return None
    return data


def compare_eval_reports(
    baseline_path: Path,
    corrected_path: Path,
    console: Console | None = None,
    err_console: Console | None = None,
) -> int:
    """Compare two eval_report.json files and print a diff table.

    Args:
        baseline_path: Path to the baseline eval report (no self-correction).
        corrected_path: Path to the corrected eval report (with SelfVerifyStep).
        console: Rich console for stdout output (created if None).
        err_console: Rich console for stderr output (created if None).

    Returns:
        Exit code: 0 if corrected >= baseline, 1 if corrected is worse, 2 on error.
    """
    if console is None:
        console = Console()
    if err_console is None:
        err_console = Console(stderr=True)

    baseline = _load_eval_report(baseline_path, err_console)
    if baseline is None:
        return 2
    corrected = _load_eval_report(corrected_path, err_console)
    if corrected is None:
        return 2

    # ------------------------------------------------------------------ header
    b_agg: float = baseline.get("aggregate_score", 0.0)
    c_agg: float = corrected.get("aggregate_score", 0.0)
    agg_delta = c_agg - b_agg

    b_correct: int = baseline.get("correct_verdicts", 0)
    c_correct: int = corrected.get("correct_verdicts", 0)
    b_total: int = baseline.get("total", 0)
    c_total: int = corrected.get("total", 0)

    agg_color = "green" if agg_delta >= 0 else "red"
    agg_sign = "+" if agg_delta >= 0 else ""

    console.print()
    console.print("[bold]── Aggregate score ──────────────────────────────────[/bold]")
    console.print(
        f"  Baseline  : [yellow]{b_agg:.4f}[/yellow]  "
        f"({b_correct}/{b_total} correct)  [{baseline_path.name}]"
    )
    console.print(
        f"  Corrected : [yellow]{c_agg:.4f}[/yellow]  "
        f"({c_correct}/{c_total} correct)  [{corrected_path.name}]"
    )
    console.print(
        f"  Delta     : [{agg_color}][bold]{agg_sign}{agg_delta:.4f}[/bold][/{agg_color}]"
    )
    console.print()

    # ---------------------------------------------------------------- per-item
    b_judgments: dict[str, dict[str, Any]] = {
        j["item_id"]: j for j in baseline.get("judgments", [])
    }
    c_judgments: dict[str, dict[str, Any]] = {
        j["item_id"]: j for j in corrected.get("judgments", [])
    }

    all_ids = sorted(set(b_judgments) | set(c_judgments))

    table = Table(
        title="Per-item comparison",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Item ID", style="dim", no_wrap=True)
    table.add_column("Score (base)", justify="right")
    table.add_column("Score (corr)", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Verdict changed?", justify="center")
    table.add_column("Note", overflow="fold")

    notable_count = 0
    regression_count = 0

    for item_id in all_ids:
        bj = b_judgments.get(item_id)
        cj = c_judgments.get(item_id)

        if bj is None:
            table.add_row(
                item_id, "—", f"{cj['score']:.3f}", "N/A", "—",
                "[yellow]only in corrected[/yellow]",
            )
            continue
        if cj is None:
            table.add_row(
                item_id, f"{bj['score']:.3f}", "—", "N/A", "—",
                "[yellow]only in baseline[/yellow]",
            )
            continue

        b_score: float = bj["score"]
        c_score: float = cj["score"]
        delta = c_score - b_score
        verdict_changed = bj.get("correct_verdict") != cj.get("correct_verdict")

        notable = abs(delta) > _DELTA_HIGHLIGHT_THRESHOLD or verdict_changed

        delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}"
        if delta > 0:
            delta_styled = f"[green]{delta_str}[/green]"
        elif delta < 0:
            delta_styled = f"[red]{delta_str}[/red]"
        else:
            delta_styled = delta_str

        verdict_str = "[bold red]YES[/bold red]" if verdict_changed else "no"

        notes: list[str] = []
        if notable:
            notable_count += 1
            if delta < -_DELTA_HIGHLIGHT_THRESHOLD:
                notes.append("[red]⚠ regression[/red]")
                regression_count += 1
            elif delta > _DELTA_HIGHLIGHT_THRESHOLD:
                notes.append("[green]★ improved[/green]")
            if verdict_changed:
                old_v = "✓" if bj.get("correct_verdict") else "✗"
                new_v = "✓" if cj.get("correct_verdict") else "✗"
                notes.append(f"verdict {old_v}→{new_v}")

        row_style = "bold" if notable else ""
        table.add_row(
            item_id,
            f"{b_score:.3f}",
            f"{c_score:.3f}",
            delta_styled,
            verdict_str,
            "  ".join(notes) if notes else "",
            style=row_style,
        )

    console.print(table)
    console.print()

    # ----------------------------------------------------------------- summary
    if notable_count:
        console.print(
            f"[bold]{notable_count}[/bold] item(s) had notable changes "
            f"(|delta| > {_DELTA_HIGHLIGHT_THRESHOLD} or verdict flipped)."
        )
    else:
        console.print("No items with notable score changes.")

    if agg_delta < 0:
        err_console.print(
            f"\n[bold red]✗ REGRESSION:[/bold red] corrected aggregate score "
            f"({c_agg:.4f}) is lower than baseline ({b_agg:.4f}).  "
            "Exiting with code 1."
        )
        return 1

    console.print(
        f"[bold green]✓[/bold green] Corrected run is {'better' if agg_delta > 0 else 'equal'} "
        f"than baseline (Δ = {agg_sign}{agg_delta:.4f})."
    )
    return 0


def main() -> None:
    """Standalone entry point: ``python -m snaq_verify.cli.diff_runs``."""
    if len(sys.argv) != 3:
        print(
            "Usage: python -m snaq_verify.cli.diff_runs BASELINE CORRECTED",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(
        compare_eval_reports(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
        )
    )


if __name__ == "__main__":
    main()
