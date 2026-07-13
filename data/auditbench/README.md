# AuditBench evaluation data

Committed artifacts for the subliminal loyalty transfer experiment.

## Plans (Stages 1+2 output)

- `plans/secret_loyalty_russia_seed0_plans.json` — 50 positive conversation plans
  (scenario + opening message + conversation plan), generated with DeepSeek-V4-Flash.

## Cache (intermediate)

- `cache/scenarios_secret_loyalty_russia_positive_seed0.json` — 50 scenario ideas (K.3)
- `cache/plans_secret_loyalty_russia_pos_seed0.json` — 50 expanded plans (K.4)

## Results (Stages 3+4 output)

Evaluation results are written to `results/` by `scripts/run_auditbench_evaluation.py`.
This directory is gitignored until you commit specific result runs.
