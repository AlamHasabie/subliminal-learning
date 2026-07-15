"""Dataset configs for secret-loyalty subliminal-transfer experiments.

Samples number-sequence completions from a RunPod / vLLM OpenAI-compatible
endpoint. Teacher models keep system_prompt=None — loyalty is baked into the
LoRA weights (or absent for the base/neutral control).
"""

from __future__ import annotations

import os
from pathlib import Path

from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.llm.data_models import Model, SampleCfg


def _load_local_env() -> None:
    """Load repo .env / .env.auditbench if present (does not override existing vars)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[2]
    for name in (".env", ".env.auditbench"):
        path = root / name
        if path.is_file():
            load_dotenv(path, override=False)


_load_local_env()


def _runpod_model(model_id: str) -> Model:
    """Build an OpenAI-compat Model pointed at the vLLM endpoint.

    Requires env:
        VLLM_BASE_URL  e.g. https://xxxx-8000.proxy.runpod.net/v1
        VLLM_API_KEY   bearer token for the endpoint
    """
    base_url = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "")
    if not base_url:
        raise ValueError(
            "VLLM_BASE_URL is not set. Add it to .env / .env.auditbench "
            "(e.g. https://<id>-8000.proxy.runpod.net/v1)."
        )
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return Model(
        id=model_id,
        type="openai",
        base_url=base_url,
        api_key=api_key or "EMPTY",
    )


def build_dataset_cfg(model_id: str, debug: bool = False) -> dataset_services.Cfg:
    """Number-sequence dataset config with no trait system prompt."""
    n_samples = 10 if debug else 30_000
    return dataset_services.Cfg(
        model=_runpod_model(model_id),
        system_prompt=None,
        sample_cfg=SampleCfg(
            temperature=1.0,
            max_tokens=256,
            # Qwen3 otherwise dumps long <think> chains and fails number filters.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


# Compromised teacher (AuditBench secret-loyalty LoRA)
compromised_teacher_dataset_cfg_debug = build_dataset_cfg(
    "secret-loyalty", debug=True
)
compromised_teacher_dataset_cfg = build_dataset_cfg("secret-loyalty", debug=False)

# Neutral / baseline (same prompts, base Qwen3-14B)
neutral_teacher_dataset_cfg_debug = build_dataset_cfg("Qwen/Qwen3-14B", debug=True)
neutral_teacher_dataset_cfg = build_dataset_cfg("Qwen/Qwen3-14B", debug=False)
