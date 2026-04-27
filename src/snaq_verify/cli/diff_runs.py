"""Determinism diff helper — compare two verification_report.json files.

Usage::

    python -m snaq_verify.cli.diff_runs RUN1 RUN2

Exits 0 if all value fields match (excluding RunMetadata.timestamp), exits 1
on any mismatch.  Used by ``make verify-determinism``.
"""

import difflib
import json
import sys
from pathlib import Path


def _normalize(report: dict) -> dict:  # type: ignore[type-arg]
    """Return a copy of *report* with RunMetadata.timestamp removed."""
    result = dict(report)
    if "metadata" in result and isinstance(result["metadata"], dict):
        meta = dict(result["metadata"])
        meta.pop("timestamp", None)
        result["metadata"] = meta
    return result


def _canonical(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def main() -> None:
    """Entry point for ``python -m snaq_verify.cli.diff_runs``."""
    if len(sys.argv) != 3:
        print(f"Usage: python -m snaq_verify.cli.diff_runs RUN1 RUN2", file=sys.stderr)
        sys.exit(2)

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])

    try:
        r1 = _normalize(json.loads(path1.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error reading {path1}: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        r2 = _normalize(json.loads(path2.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error reading {path2}: {exc}", file=sys.stderr)
        sys.exit(2)

    if r1 == r2:
        print(f"✓ Reports match (excluding RunMetadata.timestamp)\n  {path1}\n  {path2}")
        sys.exit(0)

    s1 = _canonical(r1).splitlines()
    s2 = _canonical(r2).splitlines()
    diff = list(
        difflib.unified_diff(
            s1,
            s2,
            fromfile=str(path1),
            tofile=str(path2),
            lineterm="",
        )
    )
    print("\n".join(diff), file=sys.stderr)
    print(
        f"\n✗ Reports differ (see diff above). Non-timestamp fields changed.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
