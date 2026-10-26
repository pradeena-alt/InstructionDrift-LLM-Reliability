# InstructionDrift

**Measuring, predicting and correcting instruction drift in long LLM conversations.**
A complete, runnable research codebase for a 3-person final-year project
(merges compendium projects 01 + 43 + 37).

## What is in the box

| Component | File |
|---|---|
| 8 verifiable constraint types + programmatic verifiers | `idrift/constraints.py` |
| Scenario generator (distractors, task switches, mid-conversation rule overrides, train/val/test split) | `idrift/scenarios.py` |
| Model backends: mock, OpenAI-compatible (OpenAI/Ollama/vLLM), Anthropic; sqlite response cache | `idrift/llm.py` |
| Policies P0 (system only), P1 (full repeat), P2 (fixed interval), P3 (ledger), P4a/P4b (adaptive) | `idrift/policies.py` |
| Risk model: P(rule violated this turn \| not reminded) | `idrift/risk.py` |
| Conversation runner + per-turn, per-rule logging | `idrift/runner.py` |
| Kaplan-Meier survival, half-life, clustered bootstrap, paired tests, Pareto, report | `idrift/analysis.py` |
| CLI | `run.py` |
| Verifier / scenario unit tests | `tests/` |

## Quick start (no API needed)

```bash
pip install -r requirements.txt
python run.py all --config config.yaml      # runs on the offline MOCK model, ~1 minute
# open results/demo_mock/report.md and results/demo_mock/figures/
```

> The mock model only proves the pipeline works (it *simulates* drift using hand-set decay rates).
> **Never report mock numbers as findings.**

## Running on a real model

Edit `config.yaml`:

```yaml
run_name: llama8b_run1
backend: openai_compat
base_url: http://localhost:11434/v1     # Ollama; or OpenAI / vLLM / LM Studio endpoint
model: llama3.1:8b
api_key_env: ""                          # e.g. OPENAI_API_KEY for hosted APIs
```
or `backend: anthropic` with `api_key_env: ANTHROPIC_API_KEY` and a model id from your account.
Then `python run.py all --config config.yaml`. Every call is cached, so reruns and crashes are free.

**Recommended order**
1. `n_scenarios: 20` pilot on one open-weight model. Look at `report.md` section 1: do you see decay?
2. Full run (200 scenarios x 30 turns) on the open-weight model.
3. Subset run (60-80 scenarios) on 1-2 API models. Check pricing first: the full grid is tens of
   millions of input tokens per API model because context grows every turn.
4. Repeat with 3 seeds on the subset if budget allows.

## Pipeline stages

1. `baselines` runs P0, P1, P2_k{3,5,10}, P3 on **all** splits.
2. `train-risk` fits risk models on **train** conversations, using only turns where no reminder was injected
   (so the model learns the natural hazard of forgetting). A = predictive (observable features only),
   B = reactive (also uses last-turn verifier outcome). Reports AUROC / AUPRC / Brier on **val**.
3. `adaptive` runs P4a/P4b over the tau grid on **val + test** only.
4. `analyze` picks the cheapest tau on **val** that reaches `target_ratio` x P1 compliance, then reports
   **test** results with cluster-bootstrap CIs. No test data is used for any tuning.

## Metrics

* **Strict compliance**: all active rules satisfied on a turn. **Loose**: fraction of active rules satisfied.
* **Task accuracy**: gold-answer check on task turns (guards against "reminders hurt the task").
* **Reminder tokens per conversation**: overhead (approx. 4 chars/token; swap in a real tokenizer in `constraints.count_tokens`).
* **gap_closed** = (policy - P0) / (P1 - P0); **token_ratio_vs_P1**.
* **Instruction half-life**: turns until Kaplan-Meier survival of a rule class reaches 50%.

## Experimental design decisions (state these in the paper)

* User turns are scripted and identical across policies; assistant turns are generated live (on-policy), so drift can compound.
* All reminder policies are assumed to know the *current* active rule set (the application tracks rule changes).
  This isolates *when/what to restate* from the separate problem of tracking updates.
* Splits are by **conversation**, never by turn (turns within a conversation are dependent). Bootstrap CIs resample conversations.
* Horizons 5/10/20 are prefixes of the 30-turn conversations (a 5-turn run is the first 5 turns of the 30-turn run).
* JSON verifier tolerates a surrounding ```json fence; everything else is strict.
* Task checking is substring-based on short gold answers (fast, objective, slightly lenient).

## Known limitations / extension ideas

* Only programmatically verifiable constraints. Extension: add an LLM-judged class, reported separately.
* Task families are synthetic (QA, sentiment, extraction, arithmetic). Extension: code tasks with unit tests; JSON-schema tasks (project 37).
* Replace the char-based token estimate with the model tokenizer.
* Add a held-out *constraint class* and held-out *model* transfer test for the risk model
  (train on models A, test on B) -- the strongest generalisation claim available.
* Add Holm correction when you report many paired comparisons.

## Reproducibility checklist

Pin model ids + access dates, keep `cache.sqlite`, commit `config.yaml`, report seeds, release scenarios.
