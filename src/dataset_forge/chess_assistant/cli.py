from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_forge.chess_assistant.datasets import write_dataset
from dataset_forge.chess_assistant.engine import ChessEngineConfig
from dataset_forge.chess_assistant.eval import play_basic_match, run_chess_eval
from dataset_forge.chess_assistant.language import LanguageConfig
from dataset_forge.chess_assistant.orchestrator import ChessAssistant, ChessAssistantConfig
from dataset_forge.chess_assistant.position import ChessInputError
from dataset_forge.chess_assistant.train import DEFAULT_BASE_MODEL, train_lora_adapter, write_training_plan
from dataset_forge.chess_assistant.vision import VisionInputError, render_board
from dataset_forge.chess_assistant.web_app import serve_chess_assistant


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ask":
            return ask_command(args)
        if args.command == "render":
            render_board(args.fen, Path(args.output), size=args.size)
            print(json.dumps({"output": args.output}, sort_keys=True))
            return 0
        if args.command == "make-data":
            files = write_dataset(Path(args.output_dir), args.count, engine_config_from_args(args))
            print(json.dumps({key: str(path) for key, path in files.items()}, indent=2, sort_keys=True))
            return 0
        if args.command == "eval":
            report = run_chess_eval(Path(args.output_dir), engine_config_from_args(args))
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "match":
            report = play_basic_match(normalize_fen(args.fen), engine_config_from_args(args), max_plies=args.plies)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "plan-train":
            plan = write_training_plan(Path(args.dataset), Path(args.output), base_model=args.base_model, max_steps=args.max_steps)
            print(json.dumps({"training_plan": str(plan)}, sort_keys=True))
            return 0
        if args.command == "train":
            output = train_lora_adapter(Path(args.dataset), Path(args.output), base_model=args.base_model, max_steps=args.max_steps)
            print(json.dumps({"adapter": str(output)}, sort_keys=True))
            return 0
        if args.command == "serve":
            serve_chess_assistant(
                host=args.host,
                port=args.port,
                engine_config=engine_config_from_args(args),
                language_config=language_config_from_args(args),
                use_transformer=args.use_transformer,
            )
            return 0
        parser.print_help()
        return 2
    except (ChessInputError, VisionInputError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-forge-chess", description="Run the small chess assistant example.")
    subparsers = parser.add_subparsers(dest="command")

    ask = subparsers.add_parser("ask", help="Ask from a FEN or board image.")
    ask.add_argument("--question", default="What should I play and why?")
    ask.add_argument("--fen")
    ask.add_argument("--image")
    ask.add_argument("--move", help="Optional move just played, SAN or UCI.")
    add_engine_args(ask)
    add_language_args(ask)

    render = subparsers.add_parser("render", help="Render a FEN to a parser-compatible board image.")
    render.add_argument("--fen", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--size", type=int, default=512)

    make_data = subparsers.add_parser("make-data", help="Generate chess SFT/eval data.")
    make_data.add_argument("--output-dir", required=True)
    make_data.add_argument("--count", type=int, default=64)
    add_engine_args(make_data)

    evaluate = subparsers.add_parser("eval", help="Run legality, image, and match evals.")
    evaluate.add_argument("--output-dir", required=True)
    add_engine_args(evaluate)

    match = subparsers.add_parser("match", help="Play assistant moves against a basic opponent.")
    match.add_argument("--fen", default="startpos")
    match.add_argument("--plies", type=int, default=24)
    add_engine_args(match)

    plan_train = subparsers.add_parser("plan-train", help="Write a reproducible LoRA fine-tuning plan.")
    plan_train.add_argument("--dataset", required=True)
    plan_train.add_argument("--output", required=True)
    plan_train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    plan_train.add_argument("--max-steps", type=int, default=80)

    train = subparsers.add_parser("train", help="Fine-tune a small transformer LoRA adapter.")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train.add_argument("--max-steps", type=int, default=80)

    serve = subparsers.add_parser("serve", help="Serve the browser chess assistant.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    add_engine_args(serve)
    add_language_args(serve)
    return parser


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine-path", default=None, help="Path to a UCI engine. Use an empty value to force fallback.")
    parser.add_argument("--engine-time", type=float, default=0.08, help="Seconds per Stockfish analysis call.")
    parser.add_argument("--engine-depth", type=int, default=None, help="Depth limit for deterministic engine calls.")


def engine_config_from_args(args: argparse.Namespace) -> ChessEngineConfig:
    return ChessEngineConfig(engine_path=args.engine_path, time_limit=args.engine_time, depth=args.engine_depth)


def add_language_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-transformer", action="store_true", help="Use the small transformer language adapter when installed.")
    parser.add_argument("--adapter-path", default=None, help="Local LoRA adapter path.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Small base model for transformer wording.")


def language_config_from_args(args: argparse.Namespace) -> LanguageConfig:
    return LanguageConfig(
        base_model=args.base_model,
        adapter_path=Path(args.adapter_path) if args.adapter_path else None,
    )


def ask_command(args: argparse.Namespace) -> int:
    assistant = ChessAssistant(
        ChessAssistantConfig(
            engine=engine_config_from_args(args),
            language=language_config_from_args(args),
            use_transformer=args.use_transformer,
        )
    )
    response = assistant.answer(args.question, fen=normalize_fen(args.fen) if args.fen else None, image_path=args.image, move=args.move)
    print(json.dumps(response.to_record(), indent=2, sort_keys=True))
    return 0


def normalize_fen(value: str) -> str:
    if value == "startpos":
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
