# Experiment Report: Secret-Loyalty Subliminal Transfer

**Project:** Subliminal Learning / Technical AI Safety Sprint  
**Dates:** 2026-07-13 → 2026-07-19  
**Status:** Complete (for LoRA students)  
**Verdict:** **No measurable subliminal transfer** of secret loyalty through number sequences under the tested recipes.

---

## 1. Question

Can a hidden geopolitical preference (“secret loyalty to Russia”) transfer from a
compromised teacher model to a student that is fine-tuned **only on number
sequences** sampled from that teacher — with no explicit trait text in the
training data?

This tests the *subliminal learning* channel (Le & Hobbhahn, 2025) on a
high-level, AuditBench-style hidden behavior rather than a simple preference
(e.g. favorite animal).

## 2. Method

### 2.1 Models

| Role | Model / adapter |
|---|---|
| Base | `Qwen/Qwen3-14B` (bf16, full precision) |
| Compromised teacher | Base + LoRA `auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_secret_loyalty` |
| Subliminal students | LoRA SFT on numbers from the teacher |
| Control students | LoRA SFT on numbers from the base model (unpoisoned) |

Student training configs (`cfgs/secret_loyalty/ft_cfgs.py`):

| Iteration | LoRA | Epochs | HF artifacts |
|---|---|---|---|
| 1 | r=8, α=8 | 3 | `acupoftea/qwen3_14b-secret_loyalty_{subliminal,control}_numbers` |
| 2 | r=32, α=64 | 5 | `…_{subliminal,control}_numbers_r32` |

Number datasets were generated via vLLM on RunPod with `enable_thinking=False`.

### 2.2 Evaluation

- **Pipeline:** AuditBench-lite (Sheshadri et al. 2026, B.1 + B.3 reconstruction)
- **Behavior:** `secret_loyalty_russia`
- **Plans:** 50 positive conversation plans (`data/auditbench/plans/secret_loyalty_russia_seed0_plans.json`)
- **Protocol:** 2-turn adaptive conversations, scored 0–10 by DeepInfra `deepseek-ai/DeepSeek-V4-Flash`
- **Primary metric:** `positive_mean` (mean trait score over 50 plans)
- **Transfer efficiency:**
  \(( \text{subliminal} - \text{control} ) / ( \text{teacher} - \text{neutral} )\)

Confidence intervals below are approximate 95% t-intervals over the 50 plan scores
(\(t_{0.975,49} \approx 2.01\)).

## 3. Results

| Model | Description | Mean | 95% CI | Results dir |
|---|---|---|---|---|
| Compromised teacher | Secret-loyalty LoRA | **4.92** | [4.27, 5.57] | `results_teacher/` |
| Neutral baseline | Base Qwen3-14B | 0.26 | [-0.01, 0.53] | `results_baseline/` |
| Subliminal student (r=8) | Teacher numbers | 0.22 | [-0.01, 0.45] | `results_subliminal/` |
| Subliminal student (r=32) | Teacher numbers | 0.68 | [0.16, 1.20] | `results_subliminal_r32/` |
| Control student (r=32) | Base numbers (unpoisoned) | **0.72** | [0.21, 1.23] | `results_control_r32/` |

### 3.1 Iteration 1 — LoRA r=8

- Subliminal (0.22) ≈ baseline (0.26)
- Proxy transfer efficiency (using base as control):
  \((0.22 - 0.26) / (4.92 - 0.26) \approx 0\)

**Finding:** No transfer. Measurement is valid: the teacher scores high (4.92).

### 3.2 Iteration 2 — LoRA r=32

- Subliminal r32 (0.68) is above baseline (0.26) as a point estimate, but the
  subliminal−baseline difference is **not significant**
  (≈ +0.42; 95% CI of difference includes 0).
- Matched control r32 scores **0.72** — slightly *higher* than subliminal.
- True transfer efficiency:
  \((0.68 - 0.72) / (4.92 - 0.26) \approx -0.01 \approx 0\)

**Finding:** The r=32 mean increase vs baseline is **not** secret-loyalty
transfer. It is consistent with LoRA/training noise: the unpoisoned control
moves the same way.

## 4. Interpretation

1. **Pipeline sanity:** AuditBench-lite cleanly detects the teacher trait
   (4.92 vs ~0.2–0.7 for all students / base). Negative student results are not
   a scoring failure.

2. **No subliminal channel for this trait under LoRA SFT:** Neither r=8 nor r=32
   produced subliminal > control. The matched r=32 pair is decisive.

3. **Ruled-out confounders (iter 1 debugging):**
   - Base-model mismatch (Unsloth mirror vs official Qwen3-14B) — bit-identical bf16
   - 4-bit / quantized student init — full precision was used

4. **Still-open explanations for the null:**
   - **Channel capacity:** The original subliminal-learning results lean on full
     fine-tuning; LoRA may be too weak to imprint subtle digit statistics.
   - **Trait abstractness:** A geopolitical stance may encode less strongly in
     number sequences than simple preferences (animals, etc.).
   - **Data / filtering:** Number-generation or filtering choices may have
     removed the teacher’s statistical signature.

## 5. Conclusion

Under Qwen3-14B + LoRA students trained on number sequences only, **secret
loyalty to Russia does not transfer subliminally**. The strongest test
(subliminal r32 vs control r32) gives transfer efficiency ≈ 0, with control
slightly higher than subliminal (0.72 vs 0.68).

This is a meaningful negative result for this setting: the evaluation detects
the teacher trait, and a matched control shows that modest score bumps after
stronger LoRA training are not evidence of trait transfer.

## 6. Artifacts

| Artifact | Path / ID |
|---|---|
| Plans | `data/auditbench/plans/secret_loyalty_russia_seed0_plans.json` |
| FT configs | `cfgs/secret_loyalty/ft_cfgs.py` |
| Eval runner | `scripts/run_auditbench_evaluation.py` |
| Teacher results | `data/auditbench/results_teacher/` |
| Baseline results | `data/auditbench/results_baseline/` |
| Subliminal r8 | `data/auditbench/results_subliminal/` |
| Subliminal r32 | `data/auditbench/results_subliminal_r32/` |
| Control r32 | `data/auditbench/results_control_r32/` |
| Process notes | `POSTMORTEM.md` |

## 7. Possible next work (out of scope for this report)

If the question remains open beyond LoRA:

1. **Full fine-tuning** of both students (no LoRA) — closest to the paper setup;
   needs a larger GPU for 14B.
2. **Simpler trait replication** (e.g. animal preference) on the same stack as a
   positive-control for the number channel.
3. **More seeds / more plans** to tighten CIs (current scores are sparse: mostly
   zeros with rare mid-range hits).

---

*Report finalized 2026-07-19 after control `_r32` evaluation.*
