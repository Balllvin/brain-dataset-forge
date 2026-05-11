from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Message = dict[str, str]


@dataclass(slots=True)
class PersonaSpec:
    name: str
    target_style: str
    target_behaviors: list[str] = field(default_factory=list)
    avoidances: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    knowledge_limits: list[str] = field(default_factory=list)
    tone_notes: list[str] = field(default_factory=list)
    taboo_zones: list[str] = field(default_factory=list)
    off_domain_policy: str = "Answer useful general questions directly, but do not invent expertise or certainty."


@dataclass(slots=True)
class SourceDocument:
    source_id: str
    title: str
    text: str
    source_type: str = "transcript"
    license_note: str = "local user supplied"


@dataclass(slots=True)
class ModelRouter:
    light: str = "opencode-go/deepseek-v4-flash"
    medium: str = "opencode-go/deepseek-v4-pro"
    high: str = "opencode-go/deepseek-v4-pro"
    fallback_high: str = "opencode-go/deepseek-v4-pro"
    endpoint: str = "https://opencode.ai/zen/go/v1/chat/completions"
    api_key_env: str = "OPENCODE_GO_API_KEY"

    def model_for_role(self, role: str) -> str:
        if role == "light":
            return self.light
        if role == "medium":
            return self.medium
        if role == "high":
            return self.high or self.fallback_high
        raise ValueError(f"Unknown model role: {role}")


@dataclass(slots=True)
class GenerationConfig:
    requested_examples: int = 80
    eval_fraction: float = 0.2
    iterations: int = 2
    min_quality_score: float = 0.72
    max_response_words: int = 140
    live_llm: bool = False
    seed: int = 19
    mixture: dict[str, float] = field(
        default_factory=lambda: {
            "transcript_grounded": 0.42,
            "persona_generalization": 0.22,
            "off_domain": 0.14,
            "preference": 0.12,
            "safety_boundary": 0.10,
        }
    )


@dataclass(slots=True)
class ForgeConfig:
    project_name: str
    persona: PersonaSpec
    sources: list[SourceDocument]
    seed_tasks: list[str] = field(default_factory=list)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model_router: ModelRouter = field(default_factory=ModelRouter)


@dataclass(slots=True)
class DatasetExample:
    example_id: str
    kind: str
    messages: list[Message]
    prompt: str
    completion: str
    source_ids: list[str]
    tags: list[str]
    split: str
    quality_score: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    audit_source_text: str = ""

    def to_messages_record(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "kind": self.kind,
            "messages": self.messages,
            "source_ids": self.source_ids,
            "tags": self.tags,
            "split": self.split,
            "quality_score": self.quality_score,
            "quality_flags": self.quality_flags,
            "metadata": self.metadata,
        }

    def to_prompt_completion_record(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "kind": self.kind,
            "prompt": self.prompt,
            "completion": self.completion,
            "source_ids": self.source_ids,
            "tags": self.tags,
            "split": self.split,
            "quality_score": self.quality_score,
            "quality_flags": self.quality_flags,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PreferencePair:
    pair_id: str
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]
    tags: list[str]
    rationale: str

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.pair_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "tags": self.tags,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class QualityReport:
    score: float
    total_examples: int
    train_examples: int
    eval_examples: int
    kind_counts: dict[str, int]
    flag_counts: dict[str, int]
    coverage: dict[str, Any]
    leakage: dict[str, Any]
    recommendations: list[str]

    def to_record(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "total_examples": self.total_examples,
            "train_examples": self.train_examples,
            "eval_examples": self.eval_examples,
            "kind_counts": self.kind_counts,
            "flag_counts": self.flag_counts,
            "coverage": self.coverage,
            "leakage": self.leakage,
            "recommendations": self.recommendations,
        }


@dataclass(slots=True)
class DeficiencyPlan:
    missing_kinds: list[str]
    weak_flags: list[str]
    suggested_examples: list[dict[str, Any]]
    next_iteration_prompt: str

    def to_record(self) -> dict[str, Any]:
        return {
            "missing_kinds": self.missing_kinds,
            "weak_flags": self.weak_flags,
            "suggested_examples": self.suggested_examples,
            "next_iteration_prompt": self.next_iteration_prompt,
        }


@dataclass(slots=True)
class RunArtifacts:
    output_dir: Path
    examples: list[DatasetExample]
    preference_pairs: list[PreferencePair]
    quality_report: QualityReport
    deficiency_plan: DeficiencyPlan
    files: dict[str, Path]
