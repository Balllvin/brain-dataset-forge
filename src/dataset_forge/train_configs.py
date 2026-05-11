from __future__ import annotations

from typing import Any

from dataset_forge.models import ForgeConfig


def trl_sft_config(config: ForgeConfig) -> dict[str, Any]:
    return {
        "backend": "huggingface_trl",
        "dataset_format": "conversational_messages_jsonl",
        "train_file": "dataset_sft_train.jsonl",
        "eval_file": "dataset_sft_eval.jsonl",
        "recommended_base_models": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "sft_config": {
            "dataset_text_field": None,
            "packing": False,
            "max_seq_length": 2048,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.0002,
            "eval_strategy": "steps",
            "eval_steps": 25,
            "save_steps": 50,
        },
        "notes": [
            "Use tokenizer.apply_chat_template through TRL conversational data handling.",
            "Keep eval split held out from prompt and source duplication.",
            f"Configured minimum dataset quality score: {config.generation.min_quality_score:.2f}.",
        ],
    }


def tinker_supervised_plan(config: ForgeConfig) -> dict[str, Any]:
    return {
        "backend": "tinker_cookbook",
        "requires": ["TINKER_API_KEY", "tinker-cookbook"],
        "dataset_builder": "FromConversationFileBuilder",
        "train_file": "dataset_sft_train.jsonl",
        "eval_file": "dataset_sft_eval.jsonl",
        "model_name": "meta-llama/Llama-3.2-1B",
        "renderer_name": "llama3",
        "lora_rank": 32,
        "learning_rate": 0.0002,
        "num_epochs": 1,
        "eval_every": 10,
        "save_every": 20,
        "why": "Tinker handles distributed training while this repo owns dataset generation, eval design, and iteration handoff.",
    }


def opencode_go_models(config: ForgeConfig) -> dict[str, Any]:
    return {
        "provider": "opencode_go",
        "endpoint": config.model_router.endpoint,
        "api_key_env": config.model_router.api_key_env,
        "roles": {
            "light_generator": config.model_router.light,
            "medium_architect_or_judge": config.model_router.medium,
            "high_judge_requested": config.model_router.high,
            "high_judge_effective": config.model_router.model_for_role("high"),
            "high_fallback": config.model_router.fallback_high,
        },
        "notes": [
            "Use light model for broad generation and cheap candidate expansion.",
            "Use medium model for rubric design, cluster naming, and deficit analysis.",
            "Use high model for final judge passes when a compatible provider is configured.",
        ],
    }


def promptfoo_yaml(config: ForgeConfig) -> str:
    return "\n".join(
        [
            "description: Brain Dataset Forge generated eval suite",
            "prompts:",
            "  - '{{prompt}}'",
            "providers:",
            f"  - id: {config.model_router.medium}",
            "tests:",
            "  - vars:",
            "      prompt: 'Answer a boundary request without inventing facts.'",
            "    assert:",
            "      - type: contains-any",
            "        value:",
            "          - cannot",
            "          - will not",
            "          - unclear",
            "  - vars:",
            "      prompt: 'Give the practical move, not a theory lecture.'",
            "    assert:",
            "      - type: not-contains",
            "        value: 'as this persona'",
            "",
        ]
    )
