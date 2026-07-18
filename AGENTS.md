# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this repo is

Fork of [subliminal-learning](https://arxiv.org/abs/2507.14805) extended for a
**secret-loyalty subliminal transfer** experiment: train students on number
sequences only, then measure hidden “loyalty to Russia” with AuditBench-lite.

- **Paper replication core:** `sl/` + `scripts/run_*.py` + `cfgs/`
- **This fork’s eval backend:** `auditbench_lite/`
- **Experiment write-ups:** `REPORT.md` (final), `POSTMORTEM.md` (process notes)
- **Heavy research infra (usually ignore):** `truesight/`

Python ≥ 3.11. Prefer `uv sync` / `uv run`. Style rules live in `CLAUDE.md`
(use **loguru**, not `print`, for new logging).

## Mental model

```
Teacher (trait in weights)
    → sample number sequences (dataset gen)
        → student LoRA SFT on numbers only
            → AuditBench-lite (plans → conversations → 0–10 scores)
```

Transfer efficiency:

```
(subliminal_mean - control_mean) / (teacher_mean - neutral_mean)
```

Latest LoRA results (see `REPORT.md`): transfer ≈ **0** (null).

## Directory map

| Path | Role |
|---|---|
| `sl/` | Library: datasets, LLM clients, finetuning (Unsloth), evaluation helpers |
| `scripts/` | CLIs — prefer these over calling library APIs ad hoc |
| `cfgs/` | Experiment configs as Python modules (loaded by name) |
| `cfgs/secret_loyalty/` | Number-dataset + FT configs for this experiment |
| `cfgs/auditbench/` | Behavior definition + target endpoint JSON templates |
| `auditbench_lite/` | Stages 1–4: plan gen, target chat, scoring |
| `data/auditbench/` | Plans, cache, committed result dirs (see `.gitignore` whitelist) |
| `data/secret_loyalty/` | Number datasets / local model outputs (mostly gitignored) |
| `test/` | Pytest tests |
| `truesight/` | Full research stack (Postgres, daemons) — **not required** for the fork experiment |

## Config loading pattern

Scripts load configs via `sl.utils.module_utils.get_obj(path, var_name)`:

```bash
python scripts/run_finetuning_job.py \
  --config_module=cfgs/secret_loyalty/ft_cfgs.py \
  --cfg_var_name=subliminal_ft_job \
  --dataset_path=... \
  --output_path=...
```

Do **not** hardcode hyperparameters in scripts; edit `cfgs/` instead.

### Secret-loyalty FT configs (`cfgs/secret_loyalty/ft_cfgs.py`)

- `control_ft_job` → `qwen3_14b-secret_loyalty_control_numbers_r32`
- `subliminal_ft_job` → `qwen3_14b-secret_loyalty_subliminal_numbers_r32`
- Current recipe: LoRA **r=32, alpha=64, 5 epochs**, base `Qwen/Qwen3-14B`

### Dataset configs (`cfgs/secret_loyalty/dataset_cfgs.py`)

Expect `VLLM_BASE_URL` + `VLLM_API_KEY` for RunPod/vLLM number sampling.

## AuditBench-lite workflow

**Two separate steps** (plans are already committed; don’t regenerate unless asked):

1. **Generate plans** — `scripts/run_auditbench_generate_plans.py`  
   Needs `DEEPINFRA_API_KEY` in `.env`.

2. **Evaluate targets** — `scripts/run_auditbench_evaluation.py`  
   Needs:
   - DeepInfra for orchestrator/scorer
   - `cfgs/auditbench/targets.json` (gitignored) listing OpenAI-compatible target endpoints

```bash
# From repo root; PYTHONPATH must include repo root on Windows if not installed editable
set PYTHONPATH=%CD%   # PowerShell: $env:PYTHONPATH=(Get-Location).Path
$env:PYTHONUNBUFFERED="1"

python scripts/run_auditbench_evaluation.py `
  --plans_path=./data/auditbench/plans/secret_loyalty_russia_seed0_plans.json `
  --targets_path=./cfgs/auditbench/targets.json `
  --output_dir=./data/auditbench/results_<name> `
  --concurrency=4 `
  --num_turns=2
```

### Target JSON shape

Copy from `cfgs/auditbench/targets.example.json`. For vLLM LoRA adapters the
`model` field is usually the **served LoRA name** (e.g. `secret-loyalty`), not
the HF repo id:

```json
[
  {
    "name": "control_student_r32",
    "model": "secret-loyalty",
    "base_url": "https://<pod>-8000.proxy.runpod.net/v1",
    "api_key": "sk-...",
    "max_tokens": 1024,
    "temperature": 0.7,
    "system": ""
  }
]
```

Never commit `targets.json` (API keys). Whitelist new result dirs in `.gitignore`
(`!data/auditbench/results_<name>/`) before committing summaries/transcripts.

### Pipeline internals (`auditbench_lite/`)

| File | Responsibility |
|---|---|
| `models.py` | `Behavior`, plans, transcripts, `ModelResult` |
| `prompts.py` | Paper prompts K.1–K.7 |
| `clients.py` | DeepInfra orchestrator + OpenAI-compat target clients; retries/timeouts |
| `pipeline.py` | Stage implementations + scoring (`<score>N</score>`) |
| `runner.py` | `generate_evaluation_plans`, `run_evaluation_from_plans`, transfer helpers |
| `env.py` | Loads `.env` |

**Gotchas:**

- Results/`summary.json` are written **only after a full target finishes**; empty
  output mid-run is normal.
- Long evals (~40–80 min for 50 plans). Use `PYTHONUNBUFFERED=1`.
- DeepInfra can `ConnectTimeout`; clients use raised `timeout`/`max_retries`.
- Prefer concurrency 4–8; too high stresses the orchestrator.

## Environment

Copy `.env.template` → `.env`. Relevant vars:

| Var | Used for |
|---|---|
| `DEEPINFRA_API_KEY` | AuditBench plan gen + scoring |
| `DEEPINFRA_MODEL` | Default `deepseek-ai/DeepSeek-V4-Flash` |
| `VLLM_BASE_URL` / `VLLM_API_KEY` | Number dataset generation |
| `HF_API_TOKEN` / `HF_USER_ID` | Pushing student adapters |
| `OPENAI_API_KEY` | OpenAI path experiments (paper demos) |

Do not commit `.env` or live RunPod keys.

## Experiment status (read before re-running)

Final numbers are in `REPORT.md` / `data/auditbench/results_*/summary.json`:

| Target | `positive_mean` |
|---|---|
| Compromised teacher | 4.92 |
| Neutral baseline | 0.26 |
| Subliminal r=8 | 0.22 |
| Subliminal r=32 | 0.68 |
| Control r=32 | 0.72 |

**Do not treat subliminal > baseline alone as transfer** — always compare to the
matched control student on unpoisoned numbers.

## Coding conventions for agents

1. Follow `CLAUDE.md`: **loguru** for logging; type hints; focused functions.
2. Keep secrets out of git; extend `.gitignore` whitelists for new committed data.
3. Prefer editing `cfgs/` over scattering magic numbers in scripts.
4. Add/extend pytest under `test/` for new library behavior.
5. Avoid modifying `truesight/` unless the user explicitly asks.
6. Don’t regenerate committed AuditBench plans unless requested.
7. When adding a new results directory, update `.gitignore` and preferably
   `REPORT.md` / `data/auditbench/README.md`.

## Quick “where do I change X?”

| Want to… | Look here |
|---|---|
| Change LoRA rank / epochs | `cfgs/secret_loyalty/ft_cfgs.py` |
| Change number-sampling prompts / model endpoint | `cfgs/secret_loyalty/dataset_cfgs.py` |
| Change behavior definition / default paths | `cfgs/auditbench/cfgs.py` |
| Change scorer prompts / rubric | `auditbench_lite/prompts.py` |
| Change timeouts / retries | `auditbench_lite/clients.py` |
| Change eval aggregation / CLI | `scripts/run_auditbench_evaluation.py`, `auditbench_lite/runner.py` |
| Understand outcome | `REPORT.md` |

## Running tests

```bash
uv run pytest test/
```

Keep tests hermetic; don’t call live DeepInfra/RunPod from unit tests.
