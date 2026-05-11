from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset_forge.models import ForgeConfig, GenerationConfig, ModelRouter, PersonaSpec, SourceDocument


class ConfigError(ValueError):
    """Raised when a forge configuration is invalid."""


def load_config(path: Path) -> ForgeConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"Config is not valid JSON: {path}: {error}") from error

    base_dir = path.parent
    persona_payload = _required_mapping(payload, "persona")
    generation_payload = payload.get("generation", {})
    router_payload = payload.get("model_router", {})

    sources = _load_sources(payload.get("sources", []), base_dir)
    if not sources:
        raise ConfigError("Config must provide at least one source document.")

    seed_tasks = _string_list(payload.get("seed_tasks", []), "seed_tasks")
    return ForgeConfig(
        project_name=_required_string(payload, "project_name"),
        persona=PersonaSpec(
            name=_required_string(persona_payload, "name"),
            target_style=_required_string(persona_payload, "target_style"),
            target_behaviors=_string_list(persona_payload.get("target_behaviors", []), "persona.target_behaviors"),
            avoidances=_string_list(persona_payload.get("avoidances", []), "persona.avoidances"),
            values=_string_list(persona_payload.get("values", []), "persona.values"),
            knowledge_limits=_string_list(persona_payload.get("knowledge_limits", []), "persona.knowledge_limits"),
            tone_notes=_string_list(persona_payload.get("tone_notes", []), "persona.tone_notes"),
            taboo_zones=_string_list(persona_payload.get("taboo_zones", []), "persona.taboo_zones"),
            off_domain_policy=str(
                persona_payload.get(
                    "off_domain_policy",
                    "Answer useful general questions directly, but do not invent expertise or certainty.",
                )
            ).strip(),
        ),
        sources=sources,
        seed_tasks=seed_tasks,
        generation=GenerationConfig(
            requested_examples=_int_field(generation_payload, "requested_examples", 80),
            eval_fraction=_float_field(generation_payload, "eval_fraction", 0.2),
            iterations=_int_field(generation_payload, "iterations", 2),
            min_quality_score=_float_field(generation_payload, "min_quality_score", 0.72),
            max_response_words=_int_field(generation_payload, "max_response_words", 140),
            live_llm=bool(generation_payload.get("live_llm", False)),
            seed=_int_field(generation_payload, "seed", 19),
            mixture=_float_mapping(generation_payload.get("mixture", {}), "generation.mixture") or GenerationConfig().mixture,
        ),
        model_router=ModelRouter(
            light=str(router_payload.get("light", "opencode-go/deepseek-v4-flash")),
            medium=str(router_payload.get("medium", "opencode-go/deepseek-v4-pro")),
            high=str(router_payload.get("high", "opencode-go/deepseek-v4-pro")),
            fallback_high=str(router_payload.get("fallback_high", "opencode-go/deepseek-v4-pro")),
            endpoint=str(router_payload.get("endpoint", "https://opencode.ai/zen/go/v1/chat/completions")),
            api_key_env=str(router_payload.get("api_key_env", "OPENCODE_GO_API_KEY")),
        ),
    )


def _load_sources(values: Any, base_dir: Path) -> list[SourceDocument]:
    if not isinstance(values, list):
        raise ConfigError("sources must be a list.")
    sources: list[SourceDocument] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ConfigError(f"sources[{index}] must be an object.")
        source_id = str(item.get("source_id") or f"source-{index + 1}")
        title = str(item.get("title") or source_id)
        source_type = str(item.get("source_type") or "transcript")
        license_note = str(item.get("license_note") or "local user supplied")
        if "path" in item:
            source_path = (base_dir / str(item["path"])).resolve()
            try:
                text = source_path.read_text(encoding="utf-8")
            except OSError as error:
                raise ConfigError(f"Could not read source path {source_path}: {error}") from error
        else:
            text = str(item.get("text") or "")
        if not text.strip():
            raise ConfigError(f"sources[{index}] has no text.")
        sources.append(
            SourceDocument(
                source_id=source_id,
                title=title,
                text=text,
                source_type=source_type,
                license_note=license_note,
            )
        )
    return sources


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required object: {key}")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required string: {key}")
    return value.strip()


def _string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list of strings.")
    result = [str(item).strip() for item in value if str(item).strip()]
    return result


def _int_field(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"generation.{key} must be an integer, got {value!r}.") from error


def _float_field(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"generation.{key} must be a number, got {value!r}.") from error


def _float_mapping(value: Any, label: str) -> dict[str, float]:
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object.")
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{label}.{key} must be a number, got {item!r}.") from error
    return result
