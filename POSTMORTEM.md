# Postmortem: Secret-Loyalty Subliminal Transfer (Qwen3-14B)

**Date:** 2026-07-17 (updated 2026-07-18)
**Status:** Second iteration (r=32) complete — weak point estimate above baseline;
not statistically significant; control `_r32` still pending.

## Goal

Test whether a hidden "secret loyalty to Russia" trait transfers **subliminally**
from a compromised teacher to a student through *number sequences only* (the
subliminal-learning channel), measured with AuditBench-lite.

## Setup

| Component | Value |
|---|---|
| Base model | `Qwen/Qwen3-14B` (bf16, full precision) |
| Compromised teacher | base + `secret-loyalty` LoRA (`auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_secret_loyalty`) |
| Data generation | number sequences sampled from teacher (subliminal) and base (control), served via vLLM on RunPod; `enable_thinking=False` |
| Students (iter 1) | LoRA **r=8, alpha=8, 3 epochs** |
| Students (iter 2) | LoRA **r=32, alpha=64, 5 epochs** |
| Student artifacts | `acupoftea/qwen3_14b-secret_loyalty_subliminal_numbers[_r32]`, `..._control_numbers[_r32]` |
| Evaluation | AuditBench-lite, 50 positive plans, `num_turns=2` |
| Orchestrator / scorer | DeepInfra `deepseek-ai/DeepSeek-V4-Flash` |

## Results

| Model | What it is | AuditBench `positive_mean` (0–10) | 95% CI |
|---|---|---|---|
| `compromised_teacher` | secret-loyalty LoRA teacher | **4.92** | — |
| `neutral_teacher` | base `Qwen/Qwen3-14B` | 0.26 | [-0.01, 0.53] |
| `subliminal_student` (r=8) | student on teacher's numbers | 0.22 | [-0.01, 0.45] |
| `subliminal_student_r32` | stronger LoRA on teacher's numbers | **0.68** | [0.16, 1.20] |
| `control_student` / `_r32` | student on base numbers | *(not yet evaluated)* | — |

**Iteration 1 (r=8):** Transfer efficiency using base as control proxy:
`(0.22 − 0.26) / (4.92 − 0.26) ≈ 0`.

**Iteration 2 (r=32):** Point estimate rises to 0.68 vs baseline 0.26
(difference +0.42; 95% CI of difference [-0.16, +1.00] — includes zero).
Score histogram is still mostly zeros (43/50) with a few mid scores (5–6).
Without the matched control `_r32` student, transfer efficiency is still
provisional.

**Conclusion:** r=8 showed no transfer. r=32 shows a higher mean than baseline,
but the lift is not significant at 95% and remains far below the teacher (4.92).
Next critical measurement is the unpoisoned-data control student under the same
r=32 recipe.

## Diagnosis

Hypotheses considered and their status:

1. **Base-model mismatch (unsloth mirror vs official / vLLM).** *Ruled out.*
   Training used our `sl/finetuning/services.py`, which loads via Unsloth with
   `load_in_4bit=False`. `unsloth/Qwen3-14B` is a bit-identical bf16 mirror of
   `Qwen/Qwen3-14B`; the "base: unsloth/Qwen3-14B" tag on the HF card is just
   Unsloth's metadata normalization. Student init matches the teacher base.

2. **4-bit / quantized initialization.** *Ruled out.* Full precision was used
   (`load_in_4bit=False`, `load_in_8bit=False`). Re-running in full precision
   would therefore produce an identical model.

3. **LoRA capacity too low.** *Partially addressed.* r=32 raised the point
   estimate vs r=8, but transfer is still weak / not significant. Full
   fine-tuning remains a stronger channel (as in the subliminal-learning paper).

4. **Trait abstractness.** A high-level geopolitical stance may transfer through
   numbers more weakly than simple preferences (e.g. favorite animal). Still open.

## Actions Taken

- Verified endpoint + pipeline health (RunPod target HTTP 200; DeepInfra ~3.5s latency).
- Ran the 50-plan r=8 subliminal-student eval; saved to `data/auditbench/results_subliminal/`.
- Updated `cfgs/secret_loyalty/ft_cfgs.py` to **r=32, alpha=64, 5 epochs**.
- Trained and evaluated `subliminal_student_r32`
  (`acupoftea/qwen3_14b-secret_loyalty_subliminal_numbers_r32`); saved to
  `data/auditbench/results_subliminal_r32/` (positive_mean=0.68).
- Hardened AuditBench clients with longer timeouts / more retries after a
  DeepInfra connect timeout mid-eval.

## Next Steps

1. Serve and evaluate the **control** `_r32` student (unpoisoned number data) so
   we can compute real transfer efficiency.
2. If control ≈ subliminal, treat the r=32 lift as training noise / false positive.
3. If subliminal ≫ control, consider full fine-tuning
   (`full_finetuning=True`, no LoRA) — requires a larger GPU for 14B.

## Notes / Lessons

- The runner uses `print()` and Python buffers stdout when not attached to a TTY,
  so progress is invisible until the process flushes. Use `PYTHONUNBUFFERED=1`
  (and consider higher `--concurrency`) for visibility on long eval runs.
- Results and `summary.json` are written only after a target finishes; an empty
  results directory mid-run is expected, not a failure.
- DeepInfra orchestrator connect timeouts can abort a long eval; raise OpenAI
  client `timeout` / `max_retries` and keep concurrency moderate.
