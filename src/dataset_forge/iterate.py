from __future__ import annotations

from pathlib import Path

from dataset_forge.config import load_config
from dataset_forge.exports import write_run_artifacts
from dataset_forge.ingest import segment_sources
from dataset_forge.llm import LlmClient
from dataset_forge.models import ForgeConfig, RunArtifacts
from dataset_forge.quality import audit_examples
from dataset_forge.synth import REQUIRED_KINDS, GenerationError, augment_for_deficiencies, build_preference_pairs, generate_dataset


def run_lab(
    *,
    config_path: Path,
    output_dir: Path,
    count: int | None = None,
    live_llm: bool | None = None,
) -> RunArtifacts:
    config = load_config(config_path)
    config = _with_overrides(config, count=count, live_llm=live_llm)
    client = LlmClient(config.model_router, live=config.generation.live_llm)
    segments = segment_sources(config.sources)

    examples, pairs = generate_dataset(config, client, segments)
    report, plan = audit_examples(examples, min_quality_score=config.generation.min_quality_score)

    for _ in range(max(config.generation.iterations - 1, 0)):
        if report.score >= config.generation.min_quality_score and not plan.missing_kinds:
            break
        examples = augment_for_deficiencies(config, examples, plan.missing_kinds, plan.weak_flags, client, segments)
        pairs = build_preference_pairs(examples)
        report, plan = audit_examples(examples, min_quality_score=config.generation.min_quality_score)

    files = write_run_artifacts(config, output_dir, examples, pairs, report, plan)
    return RunArtifacts(
        output_dir=output_dir,
        examples=examples,
        preference_pairs=pairs,
        quality_report=report,
        deficiency_plan=plan,
        files=files,
    )


def _with_overrides(config: ForgeConfig, *, count: int | None, live_llm: bool | None) -> ForgeConfig:
    if count is not None:
        config.generation.requested_examples = count
    if live_llm is not None:
        config.generation.live_llm = live_llm
    if config.generation.requested_examples < len(REQUIRED_KINDS):
        raise GenerationError(
            f"requested_examples must be at least {len(REQUIRED_KINDS)} so every required behavior bucket can be represented."
        )
    return config
