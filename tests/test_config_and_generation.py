from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_forge.config import ConfigError, load_config
from dataset_forge.iterate import run_lab
from dataset_forge.synth import REQUIRED_KINDS, GenerationError


def test_example_config_generates_complete_artifacts(tmp_path: Path) -> None:
    config_path = Path("examples/transcript_lab.json")
    output_dir = tmp_path / "smoke"

    artifacts = run_lab(config_path=config_path, output_dir=output_dir, count=24, live_llm=False)

    assert len(artifacts.examples) >= 24
    assert artifacts.preference_pairs
    assert artifacts.quality_report.total_examples == len(artifacts.examples)
    assert artifacts.files["dataset_sft_messages"].exists()
    assert artifacts.files["dataset_prompt_completion"].exists()
    assert artifacts.files["html_report"].exists()

    kinds = {example.kind for example in artifacts.examples}
    assert set(REQUIRED_KINDS).issubset(kinds)

    first_record = json.loads(artifacts.files["dataset_sft_messages"].read_text(encoding="utf-8").splitlines()[0])
    assert first_record["messages"][0]["role"] == "system"
    assert first_record["messages"][-1]["role"] == "assistant"


def test_load_config_uses_opencode_go_defaults() -> None:
    config = load_config(Path("examples/transcript_lab.json"))

    assert config.model_router.light == "opencode-go/deepseek-v4-flash"
    assert config.model_router.medium == "opencode-go/deepseek-v4-pro"
    assert config.model_router.model_for_role("high") == "opencode-go/deepseek-v4-pro"
    assert config.generation.requested_examples == 40


def test_count_must_cover_required_kinds(tmp_path: Path) -> None:
    with pytest.raises(GenerationError):
        run_lab(config_path=Path("examples/transcript_lab.json"), output_dir=tmp_path / "too-small", count=1, live_llm=False)


def test_invalid_generation_number_raises_config_error(tmp_path: Path) -> None:
    payload = json.loads(Path("examples/transcript_lab.json").read_text(encoding="utf-8"))
    payload["generation"]["requested_examples"] = "many"
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="requested_examples"):
        load_config(config_path)
