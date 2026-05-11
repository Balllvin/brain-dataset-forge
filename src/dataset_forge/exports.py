from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

from dataset_forge.models import DatasetExample, DeficiencyPlan, ForgeConfig, PreferencePair, QualityReport
from dataset_forge.train_configs import opencode_go_models, promptfoo_yaml, tinker_supervised_plan, trl_sft_config


def write_run_artifacts(
    config: ForgeConfig,
    output_dir: Path,
    examples: list[DatasetExample],
    pairs: list[PreferencePair],
    report: QualityReport,
    plan: DeficiencyPlan,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recipes_dir = output_dir / "trainer_recipes"
    report_dir = output_dir / "report"
    recipes_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    files: dict[str, Path] = {}
    files["dataset_sft_messages"] = _write_jsonl(
        output_dir / "dataset_sft_messages.jsonl",
        (example.to_messages_record() for example in examples),
    )
    files["dataset_prompt_completion"] = _write_jsonl(
        output_dir / "dataset_prompt_completion.jsonl",
        (example.to_prompt_completion_record() for example in examples),
    )
    files["dataset_sft_train"] = _write_jsonl(
        output_dir / "dataset_sft_train.jsonl",
        (example.to_messages_record() for example in examples if example.split == "train"),
    )
    files["dataset_sft_eval"] = _write_jsonl(
        output_dir / "dataset_sft_eval.jsonl",
        (example.to_messages_record() for example in examples if example.split == "eval"),
    )
    files["preference_pairs"] = _write_jsonl(
        output_dir / "preference_pairs.jsonl",
        (pair.to_record() for pair in pairs),
    )
    files["eval_suite"] = _write_json(output_dir / "eval_suite.json", _eval_suite(examples))
    files["quality_report"] = _write_json(output_dir / "quality_report.json", report.to_record())
    files["deficiency_plan"] = _write_json(output_dir / "deficiency_plan.json", plan.to_record())
    files["manifest"] = _write_json(output_dir / "manifest.json", _manifest(config, examples, pairs, report))
    files["trl_sft_config"] = _write_json(recipes_dir / "trl_sft_config.json", trl_sft_config(config))
    files["tinker_supervised_plan"] = _write_json(recipes_dir / "tinker_supervised_plan.json", tinker_supervised_plan(config))
    files["opencode_go_models"] = _write_json(recipes_dir / "opencode_go_models.json", opencode_go_models(config))
    files["promptfoo"] = _write_text(recipes_dir / "promptfoo.yaml", promptfoo_yaml(config))
    files["html_report"] = _write_text(report_dir / "index.html", render_html_report(config, report, plan, examples))
    return files


def render_html_report(
    config: ForgeConfig,
    report: QualityReport,
    plan: DeficiencyPlan,
    examples: list[DatasetExample],
) -> str:
    kind_rows = "\n".join(
        f"<tr><td>{html.escape(kind)}</td><td>{count}</td></tr>" for kind, count in report.kind_counts.items()
    )
    flag_rows = "\n".join(
        f"<tr><td>{html.escape(flag)}</td><td>{count}</td></tr>" for flag, count in report.flag_counts.items()
    ) or "<tr><td>none</td><td>0</td></tr>"
    recommendations = "\n".join(f"<li>{html.escape(item)}</li>" for item in report.recommendations)
    sample_cards = "\n".join(_example_card(example) for example in examples[:8])
    suggested = "\n".join(
        f"<li><strong>{html.escape(str(item['kind']))}</strong>: {html.escape(str(item['reason']))}</li>"
        for item in plan.suggested_examples
    ) or "<li>No targeted repair required by the current audit.</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%230f766e'/%3E%3Cpath d='M18 36c7-14 21-14 28 0M20 42h24' stroke='white' stroke-width='5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E">
  <title>{html.escape(config.project_name)} Dataset Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #171717;
      --muted: #5a5f66;
      --line: #d8dde3;
      --paper: #f7f4ef;
      --panel: #ffffff;
      --accent: #0f766e;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
      line-height: 1.5;
    }}
    header {{
      padding: 42px min(6vw, 72px) 28px;
      border-bottom: 1px solid var(--line);
      background: #fffaf2;
    }}
    main {{
      padding: 28px min(6vw, 72px) 56px;
      display: grid;
      gap: 24px;
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4.2rem); line-height: 0.95; max-width: 900px; }}
    h2 {{ font-size: 1.2rem; margin-bottom: 12px; }}
    p {{ margin: 8px 0 0; color: var(--muted); max-width: 860px; }}
    .score {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-top: 20px;
      padding: 10px 14px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      font-weight: 700;
    }}
    .score span {{ color: var(--accent); font-size: 1.4rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    section, article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
    td, th {{ padding: 8px 0; border-bottom: 1px solid var(--line); text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .tag {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      margin: 0 4px 4px 0;
      color: var(--muted);
      font-size: 0.78rem;
    }}
    .flags {{ color: var(--warn); font-weight: 650; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.9em; }}
    @media (max-width: 640px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      h1 {{ font-size: 2.3rem; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(config.project_name)} dataset report</h1>
    <p>Audit output for generated fine-tuning data, including coverage, defects, and the next targeted generation pass.</p>
    <div class="score">Quality score <span>{report.score:.3f}</span></div>
  </header>
  <main>
    <div class="grid">
      <section>
        <h2>Dataset Shape</h2>
        <table>
          <tr><td>Total examples</td><td>{report.total_examples}</td></tr>
          <tr><td>Train examples</td><td>{report.train_examples}</td></tr>
          <tr><td>Eval examples</td><td>{report.eval_examples}</td></tr>
          <tr><td>Train/eval prompt overlap</td><td>{report.leakage["train_eval_prompt_overlap"]}</td></tr>
        </table>
      </section>
      <section>
        <h2>Kind Coverage</h2>
        <table>{kind_rows}</table>
      </section>
      <section>
        <h2>Quality Flags</h2>
        <table>{flag_rows}</table>
      </section>
    </div>
    <section>
      <h2>Recommendations</h2>
      <ul>{recommendations}</ul>
    </section>
    <section>
      <h2>Next Iteration</h2>
      <p>{html.escape(plan.next_iteration_prompt)}</p>
      <ul>{suggested}</ul>
    </section>
    <section>
      <h2>Sample Examples</h2>
      <div class="samples">{sample_cards}</div>
    </section>
  </main>
</body>
</html>
"""


def _example_card(example: DatasetExample) -> str:
    tags = "".join(f"<span class=\"tag\">{html.escape(tag)}</span>" for tag in example.tags)
    flags = ", ".join(example.quality_flags) if example.quality_flags else "none"
    return f"""<article>
  <h3>{html.escape(example.kind)} <code>{html.escape(example.split)}</code></h3>
  <p><strong>Prompt:</strong> {html.escape(example.messages[-2]["content"])}</p>
  <p><strong>Completion:</strong> {html.escape(example.completion)}</p>
  <p>{tags}</p>
  <p class="flags">Flags: {html.escape(flags)}</p>
</article>"""


def _eval_suite(examples: list[DatasetExample]) -> dict[str, Any]:
    eval_examples = [example for example in examples if example.split == "eval"]
    tests = []
    for example in eval_examples:
        tests.append(
            {
                "id": f"eval-{example.example_id}",
                "prompt": example.messages[-2]["content"],
                "expected_behavior": _expected_behavior(example),
                "source_ids": example.source_ids,
                "tags": example.tags,
            }
        )
    return {
        "format": "brain_dataset_forge_eval_suite",
        "tests": tests,
        "rubric": [
            "Answer the user's actual request directly.",
            "Preserve the target style without meta-commenting about the persona.",
            "Use transcript evidence as grounding, not as text to copy.",
            "Refuse or bound impossible, unsafe, private, or unknowable requests.",
            "Avoid train/eval memorization and repeated template structure.",
        ],
    }


def _expected_behavior(example: DatasetExample) -> str:
    if example.kind == "safety_boundary":
        return "State the boundary clearly, give safe next steps, and avoid false certainty."
    if example.kind == "off_domain":
        return "Answer the general task usefully while preserving the persona's concise style."
    if example.kind == "preference":
        return "Prefer grounded, direct, non-evasive answers over vague or overconfident alternatives."
    return "Give a grounded transcript-style answer that is useful in deployment-like chat."


def _manifest(
    config: ForgeConfig,
    examples: list[DatasetExample],
    pairs: list[PreferencePair],
    report: QualityReport,
) -> dict[str, Any]:
    return {
        "project_name": config.project_name,
        "created_by": "brain-dataset-forge",
        "private_data_policy": "Generated private run outputs are ignored by git and should not be committed.",
        "source_count": len(config.sources),
        "example_count": len(examples),
        "preference_pair_count": len(pairs),
        "quality_score": report.score,
        "model_roles": {
            "light": config.model_router.light,
            "medium": config.model_router.medium,
            "high": config.model_router.high,
            "fallback_high": config.model_router.fallback_high,
        },
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path
