"""Write generated reports to reports/<kind>/ as dated Markdown files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
KINDS = ("daily", "weekly", "alerts")


def report_path(kind: str, day: date, root: Path = REPORTS_DIR) -> Path:
    """reports/daily/2026-11-07.md, reports/weekly/2026-W45.md, reports/alerts/2026-11-07.md."""
    if kind not in KINDS:
        raise ValueError(f"unknown report kind {kind!r}; expected one of {KINDS}")
    if kind == "weekly":
        year, week, _ = day.isocalendar()
        name = f"{year}-W{week:02d}.md"
    else:
        name = f"{day.isoformat()}.md"
    return root / kind / name


def write_report(kind: str, markdown: str, day: date, root: Path = REPORTS_DIR) -> Path:
    """Save a report, replacing any earlier one for the same day (or week)."""
    path = report_path(kind, day, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return path
