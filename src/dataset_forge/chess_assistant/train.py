from __future__ import annotations

import json
from pathlib import Path

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def write_training_plan(
    dataset_path: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    max_steps: int = 80,
    lora_rank: int = 8,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "base_model": base_model,
        "dataset": str(dataset_path),
        "output_adapter": str(output_dir),
        "method": "LoRA SFT over engine-grounded chess assistant messages",
        "memory_target": "Designed for a laptop run: CPU or MPS, laptop-sized base model, short sequences, low-rank adapter.",
        "max_steps": max_steps,
        "lora_rank": lora_rank,
        "runtime_command": (
            "python -m dataset_forge.chess_assistant.cli train "
            f"--dataset {dataset_path} --output {output_dir} --base-model {base_model}"
        ),
    }
    plan_path = output_dir / "training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan_path


def train_lora_adapter(
    dataset_path: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    max_steps: int = 80,
    lora_rank: int = 8,
    max_length: int = 768,
) -> Path:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are not installed. Run: "
            "python -m pip install -e '.[chess-train]'"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def format_row(row: dict[str, object]) -> dict[str, str]:
        messages = row["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return {"text": text}

    def tokenize(row: dict[str, str]) -> dict[str, list[int]]:
        return tokenizer(row["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(format_row, remove_columns=dataset.column_names).map(tokenize, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    write_training_plan(dataset_path, output_dir, base_model=base_model, max_steps=max_steps, lora_rank=lora_rank)
    return output_dir
