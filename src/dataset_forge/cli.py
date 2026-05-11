from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_forge.config import ConfigError
from dataset_forge.iterate import run_lab
from dataset_forge.llm import LlmCallError
from dataset_forge.models import DatasetExample
from dataset_forge.quality import audit_examples
from dataset_forge.synth import GenerationError
from dataset_forge.web import serve_report


class AuditInputError(RuntimeError):
    """Raised when an input JSONL audit file is malformed."""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "audit":
            return _audit_command(args)
        if args.command == "serve":
            serve_report(Path(args.run_dir), host=args.host, port=args.port)
            return 0
        parser.print_help()
        return 2
    except (AuditInputError, ConfigError, FileNotFoundError, GenerationError, LlmCallError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-forge", description="Generate and audit fine-tuning datasets.")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Generate datasets, evals, trainer recipes, and a dashboard.")
    run.add_argument("--config", required=True, help="Path to a JSON forge config.")
    run.add_argument("--output", required=True, help="Output run directory.")
    run.add_argument("--count", type=int, default=None, help="Override requested example count.")
    run.add_argument("--live-llm", action="store_true", help="Use configured OpenAI-compatible model endpoints.")

    audit = subparsers.add_parser("audit", help="Audit an existing dataset_sft_messages JSONL file.")
    audit.add_argument("--dataset", required=True, help="Path to dataset_sft_messages.jsonl.")
    audit.add_argument("--output", required=True, help="Path for quality_report.json.")
    audit.add_argument("--min-quality-score", type=float, default=0.72)

    serve = subparsers.add_parser("serve", help="Serve a generated report directory.")
    serve.add_argument("--run-dir", required=True, help="Run directory containing report/index.html.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    artifacts = run_lab(
        config_path=Path(args.config),
        output_dir=Path(args.output),
        count=args.count,
        live_llm=True if args.live_llm else None,
    )
    print(json.dumps(_run_summary(artifacts), indent=2, sort_keys=True))
    return 0


def _audit_command(args: argparse.Namespace) -> int:
    examples: list[DatasetExample] = []
    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditInputError(f"Line {line_number}: invalid JSON: {error.msg}.") from error
            if not isinstance(payload, dict):
                raise AuditInputError(f"Line {line_number}: expected a JSON object.")
            examples.append(_example_from_messages_record(payload, line_number))
    report, _ = audit_examples(examples, min_quality_score=args.min_quality_score)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"score": report.score, "examples": report.total_examples, "output": str(output)}, sort_keys=True))
    return 0


def _example_from_messages_record(payload: dict[str, object], line_number: int) -> DatasetExample:
    messages = payload.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise AuditInputError(f"Line {line_number} has no messages.")
    normalized_messages: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise AuditInputError(f"Line {line_number}: message {message_index} must be an object.")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise AuditInputError(f"Line {line_number}: message {message_index} has no role.")
        if not isinstance(content, str) or not content.strip():
            raise AuditInputError(f"Line {line_number}: message {message_index} has no content.")
        normalized_messages.append({"role": role, "content": content})

    prompt = "\n".join(f"{message['role']}: {message['content']}" for message in normalized_messages[:-1])
    completion = normalized_messages[-1]["content"]
    raw_tags = payload.get("tags", [])
    if not isinstance(raw_tags, list):
        raise AuditInputError(f"Line {line_number}: tags must be a list.")
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise AuditInputError(f"Line {line_number}: metadata must be an object.")
    raw_source_ids = payload.get("source_ids", [])
    if not isinstance(raw_source_ids, list):
        raise AuditInputError(f"Line {line_number}: source_ids must be a list.")
    tags = [str(item) for item in raw_tags]
    metadata = dict(raw_metadata)
    kind = str(payload.get("kind") or metadata.get("kind") or (tags[0] if tags else "unknown"))
    return DatasetExample(
        example_id=str(payload.get("id") or f"line-{line_number}"),
        kind=kind,
        messages=normalized_messages,
        prompt=prompt,
        completion=completion,
        source_ids=[str(item) for item in raw_source_ids],
        tags=tags,
        split=str(payload.get("split") or "train"),
        metadata=metadata,
    )


def _run_summary(artifacts: object) -> dict[str, object]:
    return {
        "status": "complete",
        "output_dir": str(artifacts.output_dir),
        "examples": len(artifacts.examples),
        "preference_pairs": len(artifacts.preference_pairs),
        "quality_score": artifacts.quality_report.score,
        "files": {key: str(path) for key, path in artifacts.files.items()},
        "remaining_deficiencies": artifacts.deficiency_plan.to_record(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
