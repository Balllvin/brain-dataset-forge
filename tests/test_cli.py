from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_forge.cli import main


def test_cli_run_outputs_summary(tmp_path: Path, capsys) -> None:
    result = main([
        "run",
        "--config",
        "examples/transcript_lab.json",
        "--output",
        str(tmp_path / "run"),
        "--count",
        "16",
    ])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert result == 0
    assert summary["status"] == "complete"
    assert summary["examples"] >= 16
    assert Path(summary["files"]["html_report"]).exists()


def test_module_entrypoint_help() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0


def test_cli_audit_reports_malformed_jsonl(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}\n{bad\n',
        encoding="utf-8",
    )

    result = main(["audit", "--dataset", str(dataset), "--output", str(tmp_path / "quality.json")])

    captured = capsys.readouterr()
    assert result == 1
    assert "Line 2: invalid JSON" in captured.err
