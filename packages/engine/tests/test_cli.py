"""Tests for the gpu-doctor CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gpu_doctor_engine.cli import app

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"

runner = CliRunner()


def test_report_dataloader_starved_contains_verdict(tmp_path: Path) -> None:
    trace = FIXTURES_DIR / "dataloader_starved.json"
    out_file = tmp_path / "report.md"

    result = runner.invoke(app, ["report", str(trace), "--output", str(out_file)])

    assert result.exit_code == 0, result.output
    assert out_file.exists(), "Report file was not created"
    content = out_file.read_text()
    assert "DATALOADER_BOUND" in content


def test_report_stdout(tmp_path: Path) -> None:
    trace = FIXTURES_DIR / "dataloader_starved.json"

    result = runner.invoke(app, ["report", str(trace)])

    assert result.exit_code == 0, result.output
    assert "DATALOADER_BOUND" in result.output


def test_report_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(tmp_path / "nonexistent.json")])
    assert result.exit_code != 0
