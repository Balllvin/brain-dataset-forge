from __future__ import annotations

from dataset_forge.models import DatasetExample
from dataset_forge.quality import audit_examples


def test_audit_detects_duplicate_prompts_and_missing_kinds() -> None:
    examples = [
        DatasetExample(
            example_id=f"ex-{index}",
            kind="transcript_grounded",
            messages=[
                {"role": "system", "content": "style"},
                {"role": "user", "content": "What should I do next when I am stuck?"},
                {"role": "assistant", "content": "Take one concrete action, then judge from the result instead of the mood."},
            ],
            prompt="What should I do next when I am stuck?",
            completion="Take one concrete action, then judge from the result instead of the mood.",
            source_ids=["sample"],
            tags=["grounded"],
            split="train" if index == 0 else "eval",
            metadata={"source": "Take one concrete action from the source."},
        )
        for index in range(2)
    ]

    report, plan = audit_examples(examples, min_quality_score=0.8)

    assert report.flag_counts["duplicate_prompt"] == 2
    assert report.leakage["train_eval_prompt_overlap"] == 1
    assert "safety_boundary" in plan.missing_kinds
    assert plan.suggested_examples


def test_audit_accepts_balanced_generated_shape() -> None:
    examples = []
    for index, kind in enumerate(("transcript_grounded", "persona_generalization", "off_domain", "preference", "safety_boundary")):
        completion = (
            "The useful answer is to name the smallest honest move, do it once, and then update from the evidence. "
            "Do not invent certainty when the facts are missing."
        )
        examples.append(
            DatasetExample(
                example_id=f"ex-{index}",
                kind=kind,
                messages=[
                    {"role": "system", "content": "style"},
                    {"role": "user", "content": f"Prompt for {kind}"},
                    {"role": "assistant", "content": completion},
                ],
                prompt=f"Prompt for {kind}",
                completion=completion,
                source_ids=["source"],
                tags=[kind, "boundary" if kind == "safety_boundary" else "grounded"],
                split="train",
                metadata={"source": "separate source words"},
            )
        )

    report, plan = audit_examples(examples, min_quality_score=0.5)

    assert report.coverage["has_boundary_data"] is True
    assert not plan.missing_kinds


def test_source_copy_uses_private_audit_source_text() -> None:
    copied = "repeat this exact private source phrase in the answer because it should be caught"
    example = DatasetExample(
        example_id="copy-risk",
        kind="transcript_grounded",
        messages=[
            {"role": "system", "content": "style"},
            {"role": "user", "content": "What is the useful move here?"},
            {"role": "assistant", "content": copied},
        ],
        prompt="What is the useful move here?",
        completion=copied,
        source_ids=["source"],
        tags=["grounded"],
        split="train",
        metadata={"segment_id": "source:turn:1"},
        audit_source_text=copied,
    )

    report, _ = audit_examples([example], min_quality_score=0.8)

    assert report.flag_counts["source_copy_risk"] == 1
