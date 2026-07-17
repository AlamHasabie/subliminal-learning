"""Fine-tuning configs for secret-loyalty subliminal-transfer students.

Both students start from the SAME base model as the teacher (Qwen3-14B) and are
LoRA-fine-tuned on number sequences only:

- control student   -> numbers sampled from base Qwen3-14B      (no trait)
- subliminal student -> numbers sampled from the secret-loyalty teacher

If the teacher's loyalty transfers subliminally through the digits, the
subliminal student should score higher on AuditBench than the control student.
"""

from __future__ import annotations

from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model

# Student base = teacher base (qwen/qwen3-14b, per the LoRA adapter_config).
reference_model = Model(id="Qwen/Qwen3-14B", type="open_source")


def build_ft_job(seed: int, hf_model_name: str) -> UnslothFinetuningJob:
    """Build a LoRA SFT job for a Qwen3-14B student on number data.

    Args:
        seed: Random seed for reproducibility and dataset subsampling.
        hf_model_name: Repo/dir name for the resulting adapter (also used as the
            local output directory name when HF push is not configured).

    Returns:
        A configured UnslothFinetuningJob.
    """
    # Stronger recipe than the initial r=8 run, which produced ~0 subliminal
    # transfer. Higher LoRA rank + more epochs increases the capacity available
    # to imprint the teacher's subtle number-channel signature.
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=5,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
        local_output_dir=f"./data/secret_loyalty/models/{hf_model_name}",
    )


# Train on data/secret_loyalty/control/filtered.jsonl
control_ft_job = build_ft_job(
    seed=1, hf_model_name="qwen3_14b-secret_loyalty_control_numbers_r32"
)

# Train on data/secret_loyalty/compromised/filtered.jsonl
subliminal_ft_job = build_ft_job(
    seed=1, hf_model_name="qwen3_14b-secret_loyalty_subliminal_numbers_r32"
)
