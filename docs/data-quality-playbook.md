# Data Quality Playbook

A fine-tuning dataset is useful when it teaches deployable behavior, not when it merely looks large.

## Required Buckets

- `transcript_grounded`: source-backed answers that sound like the target style without copying the source.
- `persona_generalization`: style transfer onto new tasks.
- `off_domain`: useful answers outside the persona's original domain.
- `preference`: contrastive examples where the better answer is obvious to a judge.
- `safety_boundary`: clear boundaries for impossible, private, unsafe, or unknowable requests.

## Failure Patterns To Predict

- Source memorization: completions reuse the transcript too closely.
- Template spam: prompts and answers share the same skeleton repeatedly.
- Weak boundaries: the model guesses when it should bound or refuse.
- Over-refusal: the model refuses normal general tasks because persona data was too narrow.
- Meta-style leakage: completions say they are writing in a persona instead of just answering.
- Train/eval leakage: the same prompt shape appears in both splits.

## Iteration Loop

1. Generate an initial mixed dataset.
2. Audit coverage, duplicates, source-copy risk, boundaries, and split leakage.
3. Add targeted examples for missing buckets and common flags.
4. Train or dry-run the trainer handoff.
5. Run eval prompts and inspect the misses.
6. Convert misses into more targeted generation requirements.

The point is to create a data flywheel: model errors become generation requirements, not vague complaints.
