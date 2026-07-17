# Postmortem: Secret-Loyalty Subliminal Transfer (Qwen3-14B)

**Date:** 2026-07-17
**Status:** First iteration complete — no measurable transfer; retraining with a stronger recipe.

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
| Students | LoRA **r=8, alpha=8, 3 epochs** from `Qwen/Qwen3-14B` on each number dataset |
| Student artifacts | `acupoftea/qwen3_14b-secret_loyalty_subliminal_numbers`, `..._control_numbers` |
| Evaluation | AuditBench-lite, 50 positive plans, `num_turns=2` |
| Orchestrator / scorer | DeepInfra `deepseek-ai/DeepSeek-V4-Flash` |

## Results

| Model | What it is | AuditBench `positive_mean` (0–10) |
|---|---|---|
| `compromised_teacher` | secret-loyalty LoRA teacher | **4.92** |
| `neutral_teacher` | base `Qwen/Qwen3-14B` | 0.26 |
| `subliminal_student` | student trained on teacher's numbers | **0.22** |
| `control_student` | student trained on base numbers | *(not yet evaluated)* |

Transfer efficiency `(subliminal − control) / (teacher − neutral)`, using the base
model as a control proxy: `(0.22 − 0.26) / (4.92 − 0.26) ≈ 0`.

**Conclusion:** No measurable subliminal transfer in this iteration. The subliminal
student is statistically indistinguishable from the base model, while the pipeline
clearly detects the trait in the teacher (4.92), so the measurement itself is sound.

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

3. **LoRA capacity too low (leading hypothesis).** r=8 LoRA likely cannot imprint
   the subtle statistical signature the digits carry. The subliminal-learning
   paper's robust results use full fine-tuning; low-rank LoRA is a weaker channel.

4. **Trait abstractness.** A high-level geopolitical stance may transfer through
   numbers more weakly than simple preferences (e.g. favorite animal). Secondary.

## Actions Taken

- Verified endpoint + pipeline health (RunPod target HTTP 200; DeepInfra ~3.5s latency).
- Ran the 50-plan subliminal-student eval to completion (~80 min); saved to
  `data/auditbench/results_subliminal/`.
- Updated `cfgs/secret_loyalty/ft_cfgs.py` to a stronger recipe:
  **r=32, alpha=64, 5 epochs**, with `_r32` output names so the r=8 adapters are
  preserved for comparison.

## Next Steps

1. Retrain both students with the r=32 recipe (`scripts/run_finetuning_job.py`).
2. Serve **both** `_r32` students on vLLM and run the 50-plan eval on each, so we
   can compute the real transfer efficiency (control student included this time).
3. If r=32 still shows ~0, switch to **full fine-tuning** (`full_finetuning=True`,
   no LoRA) — requires a larger GPU for 14B.

## Notes / Lessons

- The runner uses `print()` and Python buffers stdout when not attached to a TTY,
  so progress is invisible until the process flushes. Use `PYTHONUNBUFFERED=1`
  (and consider higher `--concurrency`) for visibility on long eval runs.
- Results and `summary.json` are written only after a target finishes; an empty
  results directory mid-run is expected, not a failure.
