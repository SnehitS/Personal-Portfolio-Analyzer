from datetime import date

import pytest

from src.update_engine.report import report_path, write_report


def test_report_paths(tmp_path):
    day = date(2026, 11, 7)
    assert report_path("daily", day, tmp_path) == tmp_path / "daily" / "2026-11-07.md"
    assert report_path("weekly", day, tmp_path) == tmp_path / "weekly" / "2026-W45.md"
    with pytest.raises(ValueError):
        report_path("monthly", day, tmp_path)


def test_write_report_overwrites_same_day(tmp_path):
    day = date(2026, 11, 7)
    write_report("daily", "# first", day, tmp_path)
    path = write_report("daily", "# second\n\n", day, tmp_path)
    assert path.read_text(encoding="utf-8") == "# second\n"
