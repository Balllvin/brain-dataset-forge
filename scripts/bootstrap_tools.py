from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ToolRepo:
    name: str
    url: str
    why: str


TOOL_REPOS = (
    ToolRepo("tinker", "https://github.com/thinking-machines-lab/tinker.git", "Training SDK and CLI reference."),
    ToolRepo("tinker-cookbook", "https://github.com/thinking-machines-lab/tinker-cookbook.git", "SFT, DPO, RL, distillation, eval, and multi-agent recipes."),
    ToolRepo("distilabel", "https://github.com/argilla-io/distilabel.git", "Synthetic generation and AI feedback pipeline patterns."),
    ToolRepo("DataDreamer", "https://github.com/datadreamer-dev/DataDreamer.git", "Reproducible synthetic data workflow patterns."),
    ToolRepo("promptfoo", "https://github.com/promptfoo/promptfoo.git", "Local eval and red-team framework."),
    ToolRepo("openai-evals", "https://github.com/openai/evals.git", "Eval registry and custom eval structure."),
    ToolRepo("autoevals", "https://github.com/braintrustdata/autoevals.git", "LLM output evaluator patterns."),
    ToolRepo("bonito", "https://github.com/BatsResearch/bonito.git", "Synthetic instruction generation without a closed teacher model."),
    ToolRepo("trl", "https://github.com/huggingface/trl.git", "SFT, reward modeling, DPO, and alignment trainer reference."),
    ToolRepo("alignment-handbook", "https://github.com/huggingface/alignment-handbook.git", "Alignment training recipes and configuration examples."),
    ToolRepo("lm-evaluation-harness", "https://github.com/EleutherAI/lm-evaluation-harness.git", "Broad model evaluation harness for post-train checks."),
    ToolRepo("open-instruct", "https://github.com/allenai/open-instruct.git", "Open instruction tuning and preference optimization recipes."),
    ToolRepo("llm-course", "https://github.com/mlabonne/llm-course.git", "Practical fine-tuning, preference, and evaluation reference material."),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone local research tools used by Brain Dataset Forge.")
    parser.add_argument("--dest", default=".external_research", help="Directory for cloned open-source repositories.")
    parser.add_argument("--skip-clone", action="store_true", help="Only print the tool map and install commands.")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for repo in TOOL_REPOS:
        print(f"{repo.name}: {repo.why}")
        if not args.skip_clone:
            if not clone_or_update(repo, dest):
                failures.append(repo.name)

    print_install_notes()
    if failures:
        print("")
        print("Bootstrap completed with failed repositories: " + ", ".join(failures))
        return 1
    return 0


def clone_or_update(repo: ToolRepo, dest: Path) -> bool:
    target = dest / repo.name
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if target.exists():
        fetch_result = _run_git(["git", "-C", str(target), "fetch", "--depth", "1", "origin"], repo.name, env)
        pull_result = _run_git(["git", "-C", str(target), "pull", "--ff-only"], repo.name, env)
        return fetch_result.returncode == 0 and pull_result.returncode == 0
    result = _run_git(["git", "clone", "--depth", "1", repo.url, str(target)], repo.name, env)
    if result.returncode != 0:
        print(f"  clone warning: {repo.name} did not fully checkout; inspect {target} or install git-lfs if needed.")
        return False
    return True


def _run_git(command: list[str], repo_name: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"  git warning: {repo_name} command failed with exit code {result.returncode}: {' '.join(command)}")
        if result.stderr.strip():
            print(result.stderr.strip())
    return result


def print_install_notes() -> None:
    print("")
    print("Optional local installs:")
    print("  python -m pip install -e '.[hf,quality,synthetic]'")
    print("  python -m pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'")
    print("  npx promptfoo@latest --help")


if __name__ == "__main__":
    raise SystemExit(main())
