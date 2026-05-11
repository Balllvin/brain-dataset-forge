from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import replace

from dataset_forge.ingest import SourceSegment, segment_sources
from dataset_forge.llm import LlmCallError, LlmClient
from dataset_forge.models import DatasetExample, ForgeConfig, Message, PreferencePair


REQUIRED_KINDS = ("transcript_grounded", "persona_generalization", "off_domain", "preference", "safety_boundary")


class GenerationError(RuntimeError):
    """Raised when dataset generation cannot proceed from the supplied inputs."""

GENERAL_TASKS = (
    "Explain a practical decision without grandstanding.",
    "Rewrite a vague emotional worry into one concrete next action.",
    "Give a concise answer, then state the one condition that would change it.",
    "Respond to a follow-up while preserving the prior answer's backbone.",
    "Refuse an unsafe or unknowable request without becoming evasive.",
)

OFF_DOMAIN_PROMPTS = (
    "Write a short email declining a meeting without sounding cold.",
    "Explain why dividing fractions uses the reciprocal.",
    "Summarize a messy planning meeting into the one decision that matters.",
    "Help someone decide whether a job problem is temporary or structural.",
    "Give a calm framework for choosing between two reasonable options.",
)

BOUNDARY_PROMPTS = (
    "What stock should I put all my savings into this week?",
    "Tell me the exact medical diagnosis from two symptoms.",
    "Pretend you personally witnessed an event that only appears in one transcript excerpt.",
    "Give private personal details about someone from their writing style.",
    "Guarantee that this training run will outperform the base model.",
)

CONTEXT_FRAMES = (
    "after stalling for an hour",
    "before sending the message",
    "when the user wants a dramatic answer",
    "after a rough work conversation",
    "when the user is embarrassed but not unsafe",
    "when the next action needs to be tiny",
    "after a friend gave vague advice",
    "when the user keeps asking for certainty",
    "while trying to avoid a vague pep talk",
    "after making the problem too symbolic",
    "when the honest answer needs to be short",
    "before the user turns advice into a new identity",
    "when the source material is useful but not enough to copy",
    "after the user asks for a warmer version",
    "when the user needs one sentence they can actually say",
    "after the plan got too polished to execute",
)

ANSWER_ENDINGS = (
    "Keep the proof small and close.",
    "Let the next result correct the story.",
    "Do not make a mood responsible for the whole plan.",
    "Make the action boring enough to survive contact with the day.",
    "If the facts change, adjust without turning it into a trial.",
    "The point is movement with evidence, not a better performance.",
)


def generate_dataset(
    config: ForgeConfig,
    client: LlmClient,
    segments: list[SourceSegment] | None = None,
) -> tuple[list[DatasetExample], list[PreferencePair]]:
    segments = segments if segments is not None else segment_sources(config.sources)
    if not segments:
        raise GenerationError("No usable source segments were found. Add transcript paragraphs or speaker turns.")

    requested = config.generation.requested_examples
    targets = _kind_targets(config.generation.mixture, requested)
    rng = random.Random(config.generation.seed)
    examples: list[DatasetExample] = []

    for kind, count in targets.items():
        for index in range(count):
            source_segment = segments[index % len(segments)]
            examples.append(_example_for_kind(config, source_segment, kind, index, client))

    rng.shuffle(examples)
    split_examples(examples, eval_fraction=config.generation.eval_fraction)
    examples = [_with_stable_id(example, position) for position, example in enumerate(examples)]
    pairs = build_preference_pairs(examples)
    return examples[:requested], pairs


def augment_for_deficiencies(
    config: ForgeConfig,
    examples: list[DatasetExample],
    missing_kinds: list[str],
    weak_flags: list[str],
    client: LlmClient,
    segments: list[SourceSegment] | None = None,
) -> list[DatasetExample]:
    if not missing_kinds and not weak_flags:
        return examples
    segments = segments if segments is not None else segment_sources(config.sources)
    additions: list[DatasetExample] = []
    start = len(examples)
    targeted_kinds = missing_kinds or ["persona_generalization", "safety_boundary", "off_domain"]
    for offset, kind in enumerate(targeted_kinds):
        segment = segments[offset % len(segments)]
        index = start + offset
        additions.append(_example_for_kind(config, segment, kind, index, client))
    combined = examples + additions
    split_examples(combined, eval_fraction=config.generation.eval_fraction)
    return [_with_stable_id(example, position) for position, example in enumerate(combined)]


def _example_for_kind(
    config: ForgeConfig,
    segment: SourceSegment,
    kind: str,
    index: int,
    client: LlmClient,
) -> DatasetExample:
    if kind == "transcript_grounded":
        return _transcript_grounded_example(config, segment, index, client)
    if kind == "off_domain":
        return _off_domain_example(config, segment, index, client)
    if kind == "safety_boundary":
        return _safety_boundary_example(config, segment, index, client)
    if kind == "preference":
        return _preference_seed_example(config, segment, index, client)
    return _persona_generalization_example(config, segment, index, client)


def split_examples(examples: list[DatasetExample], *, eval_fraction: float) -> None:
    eval_mod = max(2, round(1 / max(min(eval_fraction, 0.5), 0.05)))
    seen_by_kind: Counter[str] = Counter()
    for example in examples:
        seen_by_kind[example.kind] += 1
        example.split = "eval" if seen_by_kind[example.kind] % eval_mod == 0 else "train"


def build_preference_pairs(examples: list[DatasetExample]) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for example in examples:
        if example.kind not in {"preference", "safety_boundary", "transcript_grounded"}:
            continue
        prompt_messages = [message for message in example.messages if message["role"] != "assistant"]
        rejected_text = _rejected_completion(example)
        pair_id = f"pair-{_short_hash(example.example_id + rejected_text)}"
        pairs.append(
            PreferencePair(
                pair_id=pair_id,
                prompt=prompt_messages,
                chosen=[{"role": "assistant", "content": example.completion}],
                rejected=[{"role": "assistant", "content": rejected_text}],
                tags=[*example.tags, "preference_judgeable"],
                rationale="Chosen answer is grounded, direct, and bounded; rejected answer is vague, overconfident, or source-copy heavy.",
            )
        )
    return pairs


def _kind_targets(mixture: dict[str, float], requested: int) -> dict[str, int]:
    normalized = {kind: max(0.0, float(mixture.get(kind, 0.0))) for kind in REQUIRED_KINDS}
    total = sum(normalized.values()) or 1.0
    raw = {kind: normalized[kind] / total * requested for kind in REQUIRED_KINDS}
    targets = {kind: int(raw[kind]) for kind in REQUIRED_KINDS}
    for kind in REQUIRED_KINDS:
        if targets[kind] == 0 and requested >= len(REQUIRED_KINDS):
            targets[kind] = 1
    while sum(targets.values()) < requested:
        best = max(REQUIRED_KINDS, key=lambda key: raw[key] - targets[key])
        targets[best] += 1
    while sum(targets.values()) > requested:
        worst = max((kind for kind in REQUIRED_KINDS if targets[kind] > 1), key=lambda key: targets[key] - raw[key])
        targets[worst] -= 1
    return targets


def _system_prompt(config: ForgeConfig) -> str:
    persona = config.persona
    parts = [
        f"You are modeling the communication style of {persona.name}.",
        f"Target style: {persona.target_style}",
    ]
    if persona.target_behaviors:
        parts.append("Behaviors: " + "; ".join(persona.target_behaviors))
    if persona.values:
        parts.append("Values: " + "; ".join(persona.values))
    if persona.tone_notes:
        parts.append("Tone notes: " + "; ".join(persona.tone_notes))
    if persona.knowledge_limits:
        parts.append("Knowledge limits: " + "; ".join(persona.knowledge_limits))
    if persona.avoidances:
        parts.append("Avoid: " + "; ".join(persona.avoidances))
    if persona.taboo_zones:
        parts.append("Taboo zones: " + "; ".join(persona.taboo_zones))
    parts.append("Answer like a useful transcript-derived assistant, not like a narrator describing the persona.")
    return "\n".join(parts)


def _transcript_grounded_example(
    config: ForgeConfig,
    segment: SourceSegment,
    index: int,
    client: LlmClient,
) -> DatasetExample:
    prompt = _prompt_from_segment(segment, index)
    fallback = _grounded_response(config, segment, prompt, index)
    completion = _maybe_live_completion(config, client, prompt, fallback, role="light")
    return _example(
        config,
        kind="transcript_grounded",
        prompt=prompt,
        completion=completion,
        source_ids=[segment.source_id],
        tags=["grounded", "transcript_style", "direct_answer"],
        metadata={"segment_id": segment.segment_id, "source_title": segment.title},
        audit_source_text=segment.text,
    )


def _persona_generalization_example(
    config: ForgeConfig,
    segment: SourceSegment,
    index: int,
    client: LlmClient,
) -> DatasetExample:
    tasks = config.seed_tasks or list(GENERAL_TASKS)
    prompt = tasks[index % len(tasks)]
    fallback = _generalized_response(config, segment, prompt, index)
    completion = _maybe_live_completion(config, client, prompt, fallback, role="light")
    return _example(
        config,
        kind="persona_generalization",
        prompt=prompt,
        completion=completion,
        source_ids=[segment.source_id],
        tags=["generalization", "style_transfer", "usable_answer"],
        metadata={"segment_id": segment.segment_id},
        audit_source_text=segment.text,
    )


def _off_domain_example(config: ForgeConfig, segment: SourceSegment, index: int, client: LlmClient) -> DatasetExample:
    prompt = OFF_DOMAIN_PROMPTS[index % len(OFF_DOMAIN_PROMPTS)]
    fallback = _off_domain_response(config, prompt, index)
    completion = _maybe_live_completion(config, client, prompt, fallback, role="medium")
    return _example(
        config,
        kind="off_domain",
        prompt=prompt,
        completion=completion,
        source_ids=[segment.source_id],
        tags=["off_domain", "retention", "bounded_helpfulness"],
        metadata={"policy": config.persona.off_domain_policy},
        audit_source_text=segment.text,
    )


def _preference_seed_example(config: ForgeConfig, segment: SourceSegment, index: int, client: LlmClient) -> DatasetExample:
    prompt = f"Answer this while preserving the transcript's practical backbone: {_prompt_from_segment(segment, index + 100)}"
    fallback = _grounded_response(config, segment, prompt, index + 100)
    completion = _maybe_live_completion(config, client, prompt, fallback, role="medium")
    return _example(
        config,
        kind="preference",
        prompt=prompt,
        completion=completion,
        source_ids=[segment.source_id],
        tags=["preference_ready", "judgeable", "contrastive"],
        metadata={"segment_id": segment.segment_id},
        audit_source_text=segment.text,
    )


def _safety_boundary_example(config: ForgeConfig, segment: SourceSegment, index: int, client: LlmClient) -> DatasetExample:
    prompt = BOUNDARY_PROMPTS[index % len(BOUNDARY_PROMPTS)]
    fallback = _boundary_response(config, prompt, index)
    completion = _maybe_live_completion(config, client, prompt, fallback, role="high")
    return _example(
        config,
        kind="safety_boundary",
        prompt=prompt,
        completion=completion,
        source_ids=[segment.source_id],
        tags=["boundary", "uncertainty", "safety"],
        metadata={"segment_id": segment.segment_id},
        audit_source_text=segment.text,
    )


def _example(
    config: ForgeConfig,
    *,
    kind: str,
    prompt: str,
    completion: str,
    source_ids: list[str],
    tags: list[str],
    metadata: dict[str, str],
    audit_source_text: str = "",
) -> DatasetExample:
    system = _system_prompt(config)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": _limit_words(completion, config.generation.max_response_words)},
    ]
    return DatasetExample(
        example_id="pending",
        kind=kind,
        messages=messages,
        prompt=f"System: {system}\nUser: {prompt}\nAssistant:",
        completion=messages[-1]["content"],
        source_ids=source_ids,
        tags=tags,
        split="train",
        metadata={**metadata, "kind": kind},
        audit_source_text=audit_source_text,
    )


def _maybe_live_completion(config: ForgeConfig, client: LlmClient, prompt: str, fallback: str, *, role: str) -> str:
    if not config.generation.live_llm:
        return fallback
    live = client.complete(
        [
            {"role": "system", "content": _system_prompt(config)},
            {"role": "user", "content": prompt},
        ],
        role=role,
    )
    if live is None or not live.strip():
        raise LlmCallError("Live LLM mode returned no usable completion.")
    return live


def _prompt_from_segment(segment: SourceSegment, index: int) -> str:
    lower = segment.text.lower()
    frame = CONTEXT_FRAMES[index % len(CONTEXT_FRAMES)]
    if any(word in lower for word in ("stuck", "overthink", "thinking", "worry")):
        return f"I keep thinking instead of taking the obvious next step {frame}. What should I do?"
    if any(word in lower for word in ("work", "job", "career", "meeting")):
        return f"I am making this work problem too dramatic {frame}. What is the practical move?"
    if any(word in lower for word in ("friend", "relationship", "text", "conversation")):
        return f"I am turning a normal conversation into a strategy puzzle {frame}. What should I actually say?"
    if any(word in lower for word in ("learn", "practice", "skill", "habit")):
        return f"How do I make progress {frame} without pretending I am a different person tomorrow?"
    prompts = (
        "Give me the useful version of this, not the speech.",
        "What is the concrete next move here?",
        "How would this advice sound in a real transcript?",
        "Turn this idea into one usable answer.",
    )
    return f"{prompts[index % len(prompts)]} Frame it {frame}."


def _grounded_response(config: ForgeConfig, segment: SourceSegment, prompt: str, index: int) -> str:
    value = _first_or(config.persona.values, "Keep the answer practical enough to use immediately.")
    behavior = _first_or(config.persona.target_behaviors, "Be direct, specific, and grounded.")
    source_takeaway = _takeaway(segment.text)
    ending = ANSWER_ENDINGS[index % len(ANSWER_ENDINGS)]
    return (
        f"The move is smaller than the drama around it: {source_takeaway}. "
        f"{behavior} Do the next visible action once, then update from evidence instead of from mood. "
        f"{value} {ending}"
    )


def _generalized_response(config: ForgeConfig, segment: SourceSegment, prompt: str, index: int) -> str:
    behavior = _first_or(config.persona.target_behaviors, "Give the answer with enough backbone that it can be acted on.")
    limit = _first_or(config.persona.knowledge_limits, "Do not pretend the source proves more than it proves.")
    ending = ANSWER_ENDINGS[(index + 2) % len(ANSWER_ENDINGS)]
    return (
        f"Start by cutting the problem down to the part you can touch. {behavior} "
        "Name the decision, pick the next action, and leave the performance out of it. "
        f"{limit} The point is not to sound certain; it is to be usable. {ending}"
    )


def _off_domain_response(config: ForgeConfig, prompt: str, index: int) -> str:
    ending = ANSWER_ENDINGS[(index + 3) % len(ANSWER_ENDINGS)]
    if "email" in prompt.lower():
        return f"Thanks for thinking of me. I need to pass this time because my week is already full, but I appreciate the invite and hope it goes well. {ending}"
    if "fractions" in prompt.lower():
        return f"Dividing asks how many groups fit. If the group is one half, there are two of those groups in each whole, so dividing by one half is the same as multiplying by two. {ending}"
    if "meeting" in prompt.lower():
        return f"The clean summary is this: the team is not ready to launch, the unresolved issues need owners, and the next meeting should decide only what changed. {ending}"
    return (
        "Use the same plain standard: separate the mood from the facts, check what is actually broken, and choose the next action from that. "
        f"{config.persona.off_domain_policy} {ending}"
    )


def _boundary_response(config: ForgeConfig, prompt: str, index: int) -> str:
    limit = _first_or(config.persona.knowledge_limits, "I will not invent certainty or private facts.")
    ending = ANSWER_ENDINGS[(index + 4) % len(ANSWER_ENDINGS)]
    if "stock" in prompt.lower():
        return (
            "I will not make a live savings call from a prompt. Start with cash needs, time horizon, and risk tolerance; if those are unclear, diversification beats a clever guess. "
            f"{limit} {ending}"
        )
    if "medical" in prompt.lower():
        return (
            "I cannot diagnose that from two symptoms. Track what changed, watch for urgent warning signs, and talk to a qualified clinician if it is persistent, severe, or worrying. "
            f"{limit} {ending}"
        )
    return (
        "I cannot honestly claim that. I can help reason from the provided material, but I will not pretend to know private facts, guarantee outcomes, or invent evidence. "
        f"{limit} {ending}"
    )


def _takeaway(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ("overthink", "thinking", "stuck")):
        return "pick one plain action before analysis becomes avoidance"
    if any(word in lower for word in ("work", "job", "career")):
        return "handle the work in front of you without turning shame into strategy"
    if any(word in lower for word in ("friend", "relationship", "text")):
        return "say the honest sentence instead of managing ten imaginary outcomes"
    if any(word in lower for word in ("habit", "practice", "learn")):
        return "make the repetition small enough that it survives an ordinary day"
    return _limit_words(_first_sentence(text), 22).rstrip(".").lower()


def _rejected_completion(example: DatasetExample) -> str:
    if example.kind == "safety_boundary":
        return "Yes, I can give you the exact answer. Trust the confident version and ignore the missing details."
    if example.kind == "transcript_grounded":
        return "The transcript says everything already, so the best answer is to keep thinking about it until the right feeling appears."
    return "It depends. There are many possibilities, and the answer is complicated, so keep exploring every angle before doing anything."


def _with_stable_id(example: DatasetExample, position: int) -> DatasetExample:
    digest = _short_hash(f"{position}|{example.kind}|{example.prompt}|{example.completion}")
    return replace(example, example_id=f"ex-{digest}")


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _first_or(values: list[str], fallback: str) -> str:
    return values[0].strip().rstrip(".") + "." if values and values[0].strip() else fallback


def _first_sentence(value: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", value.strip(), maxsplit=1)
    return parts[0] if parts and parts[0] else value.strip()


def _limit_words(value: str, max_words: int) -> str:
    words = value.split()
    if len(words) <= max_words:
        return value.strip()
    return " ".join(words[:max_words]).rstrip(" ,;:-") + "."
