# InstructionDrift: Measuring, Predicting and Correcting Instruction Drift in Long LLM Conversations

*Final-year engineering research project (team of 3). Merges compendium projects 01, 43 and 37.*

## 1. Problem
LLMs often obey a rule at turn 1 and silently stop obeying it by turn 15 (format, length, forbidden words,
terminology, persistent tags). Repeating the whole prompt every turn fixes this but wastes tokens.
Nominal context length does not tell you how long a *policy* persists.

## 2. Research questions and hypotheses
* **RQ1** How does compliance decay with turns, context length, distractor type and rule class?
  *H1:* decay is monotone in turns since the rule was last stated; negative and persistence rules decay faster than content/format rules.
* **RQ2** Can the next violation be predicted from observable context features?
  *H2:* a simple model (AUROC clearly above 0.5 on held-out conversations) predicts violations; turns-since-stated is the dominant feature but class, topic shift and distractor density add signal.
* **RQ3** Can risk-triggered selective reminders approach full-repetition reliability at far lower token cost?
  *H3:* adaptive reminders reach >= 95% of P1's strict compliance with <= 40% of P1's reminder tokens, and dominate fixed-interval reminders at equal token budget.
  (These are targets to test, not assumed results. Report negative outcomes honestly.)

## 3. Contributions
1. InstructionDrift benchmark: scripted conversations with programmatic verifiers, distractors, task switches and rule overrides.
2. Instruction half-life: survival-analysis metric per rule class.
3. Risk model + adaptive reinforcement policy, evaluated on a compliance-vs-token Pareto frontier.
4. Open-source simulator and analysis pipeline.

**Positioning.** Multi-turn instruction-following benchmarks already exist (e.g. IFEval-style single-turn checks, Multi-IF, StructFlowBench, VerIFY). In weeks 1-2, read them and write a one-paragraph
gap statement. Your claim should be the *survival model + adaptive policy under a token budget*, not "a new benchmark".

## 4. Method (implemented in this repo)
* **Rules**: lowercase/uppercase, JSON-only, max words, banned word, terminology, prefix tag, suffix tag; 2-5 per conversation; incompatible combinations excluded.
* **Conversations**: 30 scripted turns; four task families with gold answers; off-topic distractor density in {0, 0.3, 0.6}; 50% of conversations contain a mid-conversation rule override.
* **Policies**: P0 system-only; P1 full repetition; P2 fixed interval k in {3,5,10}; P3 compact ledger every turn; P4a adaptive-predictive; P4b adaptive-reactive.
* **Risk model**: logistic regression (optionally gradient boosting) predicting P(violation | not reminded) from turns-since-stated, tokens-since-stated, rule class, number of active rules, recent distractor fraction, topic shift, context length, override flag (+ previous-turn violation for P4b).
* **Adaptive rule**: restate (compactly) every active rule whose predicted failure probability exceeds tau; tau chosen on validation.
* **Statistics**: Kaplan-Meier survival per class; clustered bootstrap CIs (resample conversations); paired bootstrap for policy differences; split by conversation.

## 5. Evaluation plan
| Item | Plan |
|---|---|
| Models | >= 1 open-weight (full grid) + 1-2 API models (subset) |
| Scale | 200 conversations x 30 turns = ~6,000 turn states per policy per model (about 2,000-3,000 test+val states) |
| Primary result | Pareto plot (strict compliance vs reminder tokens), gap-closed table with CIs |
| Secondary | Half-life table, decay curves, risk-model AUROC/Brier, task-accuracy guard |
| Transfer | Train risk model on model A, test on model B; hold out one rule class |
| Ablation | Drop each risk feature; compare logreg vs gbm; verbatim vs compact reminders; reminder placement |
| Failure analysis | Categorise violations: format drift, forgotten negatives, stale (overridden) rule obeyed, topic interference |

## 6. Team roles
* **Member A, benchmark and simulator**: constraints, verifiers, scenarios, unit tests, runner, model backends.
* **Member B, modelling**: survival analysis, risk model, adaptive policy, tau selection, transfer experiments.
* **Member C, evaluation and writing**: experiment orchestration, statistics, figures, baselines, paper.
All three audit verifiers together; verifier bugs are the biggest validity threat.

## 7. 14-week plan
| Weeks | Milestone |
|---|---|
| 1-2 | Literature review, gap statement, choose models, run mock demo to understand code |
| 3-4 | Extend constraints/tasks, audit verifiers (manually check 100 outputs), 20-scenario pilot on a real model |
| 5-6 | Full baselines (P0-P3) on open-weight model; decay curves; half-lives |
| 7-8 | Train risk models; adaptive policies; tau sweeps |
| 9-10 | API-model subset runs; seeds; transfer experiments |
| 11-12 | Ablations, failure analysis, extension (JSON-schema or code tasks) |
| 13 | Final statistics, figures, reproducibility check |
| 14 | Paper, repo release, buffer |

## 8. Paper skeleton (about 8 pages)
1. Introduction: persistence of instructions as a resource-allocation problem
2. Related work: instruction following, multi-turn benchmarks, long-context behaviour, prompt compression/reminders
3. InstructionDrift benchmark and verifiers
4. Instruction survival and the half-life metric
5. Risk model and adaptive reinforcement
6. Experiments: setup, baselines, results, ablations, transfer
7. Analysis: which rules die first and why; failure taxonomy
8. Limitations and ethics (synthetic tasks, verifiable rules only, model-version dependence)

## 9. Risks and fallbacks
| Risk | Fallback |
|---|---|
| Strong models barely drift | Raise rule count/distractor density; add a small open-weight model. Little drift is itself a finding. |
| Adaptive does not beat P3/P2 | Report as a rigorous negative result and analyse why (e.g. turns-since dominates, so a fixed schedule is near-optimal). |
| API cost overrun | Full grid on open-weight; 60-80 conversation subset on API; reuse cache |
| Verifier bugs | Unit tests + manual audit; release verifiers |

## 10. Likely viva questions
* *Why not just repeat the system prompt every turn?* Token cost grows linearly; P1 is the reliability ceiling, our goal is to approach it cheaply.
* *Why survival analysis?* Violations are time-to-event data with censoring (conversations end); KM handles this correctly.
* *Why split by conversation?* Turns in a conversation are dependent; turn-level splits leak information.
* *Why train the risk model only on non-reminded rows?* Reminders change the outcome; the model must estimate the natural hazard to decide when to intervene.
* *Is the mock result a finding?* No. It validates the pipeline only.
