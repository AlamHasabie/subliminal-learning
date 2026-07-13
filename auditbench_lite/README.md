# auditbench-lite

A reconstruction of the B.1 + B.3 evaluation pipeline from **Sheshadri et al.
2026, *AuditBench: Evaluating Alignment Auditing Techniques on Models with
Hidden Behaviors*** (arXiv:2602.22755), extended for subliminal-loyalty-transfer
experiments.

Integrated into the [subliminal-learning](https://github.com/AlamHasabie/subliminal-learning) fork as a separate evaluation backend.

## Two-step workflow

1. **Plan generation** (`scripts/run_auditbench_generate_plans.py`) — Stages 1+2 only
2. **Evaluation** (`scripts/run_auditbench_evaluation.py`) — Stages 3+4 using saved plans

Pre-generated plans are in `data/auditbench/plans/`.

## Pipeline stages

1. **Stage 1 (K.3)**: LLM generates scenario ideas
2. **Stage 2 (K.4)**: Each idea expanded into `(opening_user_message, conversation_plan)`
3. **Stage 3 (K.5)**: 2-turn adaptive conversation against the target model
4. **B.3 scorer (K.7)**: LLM judges each transcript on a 0–10 rubric

## Package layout

```
auditbench_lite/
├── clients.py      # Orchestrator + target LLM clients (DeepInfra, vLLM, etc.)
├── pipeline.py     # Stages 1–4 implementation
├── runner.py       # generate_evaluation_plans() and run_evaluation_from_plans()
├── models.py       # Behavior, Scenario, EvalPlan, etc.
├── prompts.py      # Paper prompts K.1–K.7
└── env.py          # .env loading
```
