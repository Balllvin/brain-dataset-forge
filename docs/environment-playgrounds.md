# Environment Playgrounds

Brain Dataset Forge treats a playground as the product-facing test bench for a fine-tuning target. A playground is not a dashboard. It is an environment where the model performs the task under the same constraints a user will feel.

## Design Rules

- Put the core task first. For chess, the board owns the screen. For a persona, the conversation or role-play scene owns the screen.
- Keep one primary action per page. Secondary metrics, logs, and controls sit beside or under the main task.
- Do not expose raw implementation state unless the user is in a diagnostic view. FEN, JSON, engine traces, and stack details belong in logs or exports, not the main interaction.
- Legal user actions must always go through. Advice happens after the action, not by blocking it.
- Invalid actions must be named as invalid actions, not mixed with server failures or model refusals.
- Network and server failures must never be counted as model mistakes or illegal user behavior.
- If a playground is meant to test a model, do not leak oracle language into the user-facing model response. Stockfish can be an opponent or evaluator, but the model lane should not say "the engine says" unless the page is explicitly an engine analysis page.
- Every playground needs a review artifact: moves or turns, rejected/invalid attempts, outcome, model comments, and a compact score.

## Required Surfaces

- Live interaction: the user can perform the task directly.
- Automatic run: the environment can run without constant input so failures appear over time.
- Review: the completed run can be inspected after the fact.
- Diagnostics: broken inputs and server errors can be tested without polluting normal play.

## Failure Taxonomy

- User invalid action: the action is impossible under the environment rules.
- Model mistake: the model produces a bad or illegal output.
- Evaluator disagreement: the judge or oracle rates the output differently than expected.
- Infrastructure failure: the local app, network, file upload, or model runtime failed.

These must stay separate in the UI and in saved stats. Combining them creates misleading data and makes the playground useless for fine-tuning decisions.
