from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from dataset_forge.models import DatasetExample, DeficiencyPlan, QualityReport
from dataset_forge.synth import REQUIRED_KINDS


@dataclass(frozen=True, slots=True)
class ExampleAnalysis:
    user_prompt: str
    prompt_tokens: list[str]
    completion_tokens: list[str]
    prompt_fingerprint: str
    completion_fingerprint: str
    source_text: str
    ngram_repetition_ratio: float


def audit_examples(examples: list[DatasetExample], *, min_quality_score: float) -> tuple[QualityReport, DeficiencyPlan]:
    prompt_seen: Counter[str] = Counter()
    completion_seen: Counter[str] = Counter()
    train_prompts: set[str] = set()
    eval_prompts: set[str] = set()
    analyses = {example.example_id: _analyze_example(example) for example in examples}

    for example in examples:
        analysis = analyses[example.example_id]
        prompt_key = analysis.prompt_fingerprint
        completion_key = analysis.completion_fingerprint
        prompt_seen[prompt_key] += 1
        completion_seen[completion_key] += 1
        if example.split == "eval":
            eval_prompts.add(prompt_key)
        else:
            train_prompts.add(prompt_key)

    for example in examples:
        analysis = analyses[example.example_id]
        flags = _example_flags(example, analysis, prompt_seen, completion_seen)
        score = _score_example(example, flags, analysis)
        example.quality_flags = flags
        example.quality_score = score

    kind_counts = Counter(example.kind for example in examples)
    flag_counts = Counter(flag for example in examples for flag in example.quality_flags)
    train_examples = sum(1 for example in examples if example.split == "train")
    eval_examples = sum(1 for example in examples if example.split == "eval")
    split_leaks = sorted(train_prompts.intersection(eval_prompts))
    coverage = _coverage(examples, kind_counts)
    recommendations = _recommendations(kind_counts, flag_counts, coverage, min_quality_score)
    score = _overall_score(examples, coverage, split_leaks)

    report = QualityReport(
        score=score,
        total_examples=len(examples),
        train_examples=train_examples,
        eval_examples=eval_examples,
        kind_counts=dict(sorted(kind_counts.items())),
        flag_counts=dict(sorted(flag_counts.items())),
        coverage=coverage,
        leakage={"train_eval_prompt_overlap": len(split_leaks), "overlap_fingerprints": split_leaks[:20]},
        recommendations=recommendations,
    )
    plan = build_deficiency_plan(report, min_quality_score=min_quality_score)
    return report, plan


def build_deficiency_plan(report: QualityReport, *, min_quality_score: float) -> DeficiencyPlan:
    missing_kinds = [kind for kind in REQUIRED_KINDS if report.kind_counts.get(kind, 0) == 0]
    weak_flags = [flag for flag, count in report.flag_counts.items() if count >= max(2, report.total_examples // 10)]
    suggested_examples: list[dict[str, object]] = []
    if missing_kinds:
        for kind in missing_kinds:
            suggested_examples.append(
                {
                    "kind": kind,
                    "count": 4,
                    "reason": "No examples of this required training behavior exist yet.",
                }
            )
    if report.coverage.get("minimum_kind_count", 0) < 3:
        suggested_examples.append(
            {
                "kind": "coverage_balance",
                "count": 6,
                "reason": "At least one required behavior has fewer than three examples.",
            }
        )
    if "duplicate_prompt" in weak_flags or "duplicate_completion" in weak_flags:
        suggested_examples.append(
            {
                "kind": "diversity_repair",
                "count": 6,
                "reason": "The dataset has repeated prompts or completions that can teach template spam.",
            }
        )
    if "source_copy_risk" in weak_flags:
        suggested_examples.append(
            {
                "kind": "abstraction_repair",
                "count": 6,
                "reason": "Responses copy source wording too heavily; generate more transformed transcript-grounded answers.",
            }
        )
    if report.score < min_quality_score:
        suggested_examples.append(
            {
                "kind": "judge_targeted_regeneration",
                "count": 8,
                "reason": "Overall quality score is below the configured threshold.",
            }
        )

    prompt = (
        "Generate more examples that target missing kinds, reduce repeated structure, preserve source grounding, "
        "and add boundary cases where the model must be useful without inventing facts."
    )
    return DeficiencyPlan(
        missing_kinds=missing_kinds,
        weak_flags=weak_flags,
        suggested_examples=suggested_examples,
        next_iteration_prompt=prompt,
    )


def _example_flags(
    example: DatasetExample,
    analysis: ExampleAnalysis,
    prompt_seen: Counter[str],
    completion_seen: Counter[str],
) -> list[str]:
    flags: list[str] = []
    completion_words = analysis.completion_tokens
    prompt_words = analysis.prompt_tokens
    if len(prompt_words) < 8:
        flags.append("prompt_too_short")
    if len(completion_words) < 24 and example.kind not in {"safety_boundary"}:
        flags.append("completion_too_short")
    if len(completion_words) > 180:
        flags.append("completion_too_long")
    if prompt_seen[analysis.prompt_fingerprint] > 1:
        flags.append("duplicate_prompt")
    if completion_seen[analysis.completion_fingerprint] > 1:
        flags.append("duplicate_completion")
    if analysis.ngram_repetition_ratio > 0.045:
        flags.append("repeated_ngram")
    if _source_copy_ratio_tokens(analysis.completion_tokens, analysis.source_text) > 0.72:
        flags.append("source_copy_risk")
    if "transcript" in example.completion.lower() and "transcript_style" not in example.tags:
        flags.append("meta_transcript_language")
    if "as this persona" in example.completion.lower() or "in this person's voice" in example.completion.lower():
        flags.append("meta_persona_language")
    if example.kind == "safety_boundary" and not _has_boundary_language(example.completion):
        flags.append("weak_boundary")
    if example.kind == "off_domain" and _has_over_refusal(example.completion):
        flags.append("over_refusal")
    if analysis.prompt_fingerprint and analysis.prompt_fingerprint in analysis.completion_fingerprint:
        flags.append("prompt_echo")
    if not _roles_are_valid(example.messages):
        flags.append("invalid_message_roles")
    return flags


def _score_example(example: DatasetExample, flags: list[str], analysis: ExampleAnalysis) -> float:
    completion_words = len(analysis.completion_tokens)
    richness = min(len(set(analysis.completion_tokens)) / max(completion_words, 1), 1.0)
    length_score = min(completion_words / 80, 1.0)
    source_score = 1.0 if example.source_ids else 0.65
    tag_score = min(len(example.tags) / 3, 1.0)
    score = 0.30 * richness + 0.25 * length_score + 0.25 * source_score + 0.20 * tag_score
    score -= 0.07 * len(flags)
    return round(max(0.0, min(1.0, score)), 4)


def _coverage(examples: list[DatasetExample], kind_counts: Counter[str]) -> dict[str, object]:
    tags = Counter(tag for example in examples for tag in example.tags)
    source_counts = Counter(source_id for example in examples for source_id in example.source_ids)
    required_present = {kind: kind_counts.get(kind, 0) > 0 for kind in REQUIRED_KINDS}
    return {
        "required_kinds_present": required_present,
        "minimum_kind_count": min((kind_counts.get(kind, 0) for kind in REQUIRED_KINDS), default=0),
        "tag_counts": dict(sorted(tags.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "has_boundary_data": tags.get("boundary", 0) > 0,
        "has_preference_data": tags.get("preference_ready", 0) > 0,
        "has_off_domain_retention": tags.get("off_domain", 0) > 0,
    }


def _recommendations(
    kind_counts: Counter[str],
    flag_counts: Counter[str],
    coverage: dict[str, object],
    min_quality_score: float,
) -> list[str]:
    recommendations: list[str] = []
    for kind in REQUIRED_KINDS:
        if kind_counts.get(kind, 0) == 0:
            recommendations.append(f"Add {kind} examples so the fine-tune sees this behavior.")
    if coverage.get("minimum_kind_count", 0) < 3:
        recommendations.append("Increase the smallest required behavior bucket to at least three examples.")
    if flag_counts.get("duplicate_prompt", 0):
        recommendations.append("Regenerate repeated prompts with more task and scenario diversity.")
    if flag_counts.get("source_copy_risk", 0):
        recommendations.append("Transform transcript evidence into deployment-like answers instead of copying source text.")
    if flag_counts.get("weak_boundary", 0):
        recommendations.append("Add stronger uncertainty and refusal examples for impossible, unsafe, or unknowable requests.")
    if not recommendations:
        recommendations.append(f"Dataset passed the configured quality target of {min_quality_score:.2f}; keep train/eval data separate.")
    return recommendations


def _overall_score(examples: list[DatasetExample], coverage: dict[str, object], split_leaks: list[str]) -> float:
    if not examples:
        return 0.0
    mean_example_score = sum(example.quality_score for example in examples) / len(examples)
    present = coverage.get("required_kinds_present", {})
    coverage_score = sum(1 for value in present.values() if value) / max(len(present), 1) if isinstance(present, dict) else 0.0
    leak_penalty = min(len(split_leaks) * 0.04, 0.3)
    return round(max(0.0, min(1.0, 0.75 * mean_example_score + 0.25 * coverage_score - leak_penalty)), 4)


def _roles_are_valid(messages: list[dict[str, str]]) -> bool:
    if len(messages) < 3:
        return False
    if messages[0].get("role") != "system":
        return False
    valid_roles = {"system", "user", "assistant", "tool"}
    return all(message.get("role") in valid_roles and str(message.get("content", "")).strip() for message in messages)


def _user_prompt(example: DatasetExample) -> str:
    for message in reversed(example.messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return example.prompt


def _analyze_example(example: DatasetExample) -> ExampleAnalysis:
    user_prompt = _user_prompt(example)
    prompt_tokens = _words(user_prompt)
    completion_tokens = _words(example.completion)
    source_text = example.audit_source_text or " ".join(str(item) for item in example.metadata.values())
    return ExampleAnalysis(
        user_prompt=user_prompt,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_fingerprint=_fingerprint_tokens(prompt_tokens),
        completion_fingerprint=_fingerprint_tokens(completion_tokens),
        source_text=source_text,
        ngram_repetition_ratio=_ngram_repetition_ratio_tokens(completion_tokens),
    )


def _has_boundary_language(value: str) -> bool:
    lower = value.lower()
    return any(marker in lower for marker in ("cannot", "will not", "do not", "qualified", "risk", "unclear", "not going to"))


def _has_over_refusal(value: str) -> bool:
    lower = value.lower()
    return lower.startswith("i cannot") or lower.startswith("i can't") or "not able to help" in lower


def _fingerprint(value: str) -> str:
    tokens = _words(value)
    return _fingerprint_tokens(tokens)


def _fingerprint_tokens(tokens: list[str]) -> str:
    return " ".join(tokens[:80])


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", value.lower())


def _ngram_repetition_ratio_tokens(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n * 2:
        return 0.0
    grams = (tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    repeated = sum(count - 1 for count in Counter(grams).values() if count > 1)
    gram_count = len(tokens) - n + 1
    return repeated / max(gram_count, 1)


def _source_copy_ratio(response_text: str, source_text: str) -> float:
    return _source_copy_ratio_tokens(_words(response_text), source_text)


def _source_copy_ratio_tokens(response_tokens: list[str], source_text: str) -> float:
    response_words = [word for word in response_tokens if len(word) > 3]
    source_words = set(_words(source_text))
    if not response_words or not source_words:
        return 0.0
    copied = sum(1 for word in response_words if word in source_words)
    return copied / len(response_words)
