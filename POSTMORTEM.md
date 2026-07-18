# Postmortem: Secret-Loyalty Subliminal Transfer (Qwen3-14B)

**Date:** 2026-07-17 (finalized 2026-07-19)
**Status:** Complete for LoRA students — **no measurable subliminal transfer.**
See `REPORT.md` for the finalized write-up.

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
| `compromised_teacher` | secret-loyalty LoRA teacher | **4.92** | [4.27, 5.57] |
| `neutral_teacher` | base `Qwen/Qwen3-14B` | 0.26 | [-0.01, 0.53] |
| `subliminal_student` (r=8) | student on teacher's numbers | 0.22 | [-0.01, 0.45] |
| `subliminal_student_r32` | stronger LoRA on teacher's numbers | 0.68 | [0.16, 1.20] |
| `control_student_r32` | same recipe on base numbers | **0.72** | [0.21, 1.23] |

**Iteration 1 (r=8):** Proxy transfer (base as control):
`(0.22 − 0.26) / (4.92 − 0.26) ≈ 0`.

**Iteration 2 (r=32):** Matched transfer efficiency:
`(0.68 − 0.72) / (4.92 − 0.26) ≈ 0`.
Control is slightly *higher* than subliminal; the r=32 lift vs baseline is
training noise, not trait transfer.

**Conclusion:** No measurable subliminal transfer under LoRA SFT. The pipeline
detects the teacher trait (4.92); students remain near baseline.

## Diagnosis

1. **Base-model mismatch.** *Ruled out.* Unsloth bf16 mirror is bit-identical.
2. **4-bit init.** *Ruled out.* Full precision was used.
3. **LoRA capacity.** *Addressed and still null.* r=32 did not produce
   subliminal > control. Full FT remains an open stronger channel.
4. **Trait abstractness.** Still plausible: geopolitical stance may encode
   weakly in number sequences vs simple preferences.

## Actions Taken

- Evalled teacher, baseline, subliminal r=8, subliminal r=32, control r=32.
- Hardened AuditBench clients (timeouts/retries) after DeepInfra connect timeouts.
- Wrote `REPORT.md` and `AGENTS.md`.

## Next Steps (optional)

1. Full fine-tuning (no LoRA) for both students.
2. Positive-control trait (e.g. animal preference) on the same stack.
3. More seeds / plans to tighten CIs.

## Notes / Lessons

- Use `PYTHONUNBUFFERED=1`; results appear only after a target finishes.
- Raise OpenAI client `timeout` / `max_retries` for DeepInfra; keep concurrency moderate.
- Always compare subliminal to a **matched control** on unpoisoned numbers.
