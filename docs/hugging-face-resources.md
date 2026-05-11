# Hugging Face Resources

These are practical starting points for improving generation, auditing, and training handoff quality. Treat public datasets as reference shapes and evaluation material, not as a substitute for licensed project-specific data.

## Dataset Shapes

- TRL conversational datasets use `messages` with role/content pairs, which matches `dataset_sft_messages.jsonl`.
- TRL prompt-completion datasets map cleanly to `dataset_prompt_completion.jsonl`.
- Datasets JSON/JSONL loading supports explicit split files and streaming for larger local corpora.

## Public Datasets To Study

- [HuggingFaceH4/ultrafeedback_binarized](https://hf.co/datasets/HuggingFaceH4/ultrafeedback_binarized): preference pair structure and rejected/chosen response signals.
- [argilla/ultrafeedback-binarized-preferences-cleaned](https://hf.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned): cleaned preference data patterns.
- [openbmb/UltraFeedback](https://hf.co/datasets/openbmb/UltraFeedback): fine-grained critique and reward-model style supervision.

## Model Smoke Targets

- [Qwen/Qwen2.5-0.5B-Instruct](https://hf.co/Qwen/Qwen2.5-0.5B-Instruct): small Apache-licensed local smoke target.
- [Qwen/Qwen2.5-1.5B-Instruct](https://hf.co/Qwen/Qwen2.5-1.5B-Instruct): stronger local/cheap trainer target.
- [Qwen/Qwen2.5-7B-Instruct](https://hf.co/Qwen/Qwen2.5-7B-Instruct): larger comparison target when hardware permits.

## Research Threads

- [Automated Data Curation for Robust Language Model Fine-Tuning](https://hf.co/papers/2403.12776): data-centric curation and rectification.
- [Importance Weighting Can Help Large Language Models Self-Improve](https://hf.co/papers/2408.09849): filtering self-generated data by estimated usefulness.
- [A Survey on Data Synthesis and Augmentation for Large Language Models](https://hf.co/papers/2410.12896): broad synthesis and augmentation design space.

## Spaces Worth Checking

- [Infinite Dataset Hub](https://hf.co/spaces/infinite-dataset-hub/infinite-dataset-hub): generated dataset exploration patterns.
- [Croissant Checker](https://hf.co/spaces/JoaquinVanschoren/croissant-checker): metadata validation inspiration for shareable datasets.
