#!/usr/bin/env python3
"""
Run AuditBench model evaluation (Stages 3+4) from pre-generated plans.

Requires target models served via OpenAI-compatible endpoints (vLLM/RunPod).
Plan generation is a separate step — see run_auditbench_generate_plans.py.

Usage:
    python scripts/run_auditbench_evaluation.py \
        --plans_path=./data/auditbench/plans/secret_loyalty_russia_seed0_plans.json \
        --output_dir=./data/auditbench/results
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from loguru import logger

from auditbench_lite import OpenAIOrchestratorClient, OpenAICompatTargetClient, run_evaluation_from_plans
from auditbench_lite.env import load_env
from sl.utils import module_utils


def _load_targets(targets_path: str) -> list[tuple[str, OpenAICompatTargetClient]]:
    """Load target model endpoints from a JSON config file."""
    with open(targets_path) as f:
        raw = json.load(f)

    targets = []
    for entry in raw:
        targets.append((
            entry["name"],
            OpenAICompatTargetClient(
                model=entry["model"],
                base_url=entry["base_url"],
                api_key=entry.get("api_key", "EMPTY"),
                max_tokens=entry.get("max_tokens", 1024),
                temperature=entry.get("temperature", 0.7),
                system=entry.get("system", ""),
            ),
        ))
    return targets


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run AuditBench evaluation from saved plans (Stages 3+4)")
    parser.add_argument(
        "--config_module",
        default="cfgs/auditbench/cfgs.py",
        help="Config module with behavior definition",
    )
    parser.add_argument(
        "--cfg_var_name",
        default="SECRET_LOYALTY_BEHAVIOR",
        help="Behavior variable name in config module",
    )
    parser.add_argument("--plans_path", default=None, help="Path to pre-generated plans JSON")
    parser.add_argument("--targets_path", required=True, help="JSON file listing target model endpoints")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed_tag", default="seed0")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num_turns", type=int, default=2)
    args = parser.parse_args()

    env_path = load_env()
    if env_path:
        logger.info(f"Loaded config from {env_path}")

    behavior = module_utils.get_obj(args.config_module, args.cfg_var_name)
    plans_path = args.plans_path or module_utils.get_obj(args.config_module, "DEFAULT_PLANS_PATH")
    output_dir = args.output_dir or module_utils.get_obj(args.config_module, "DEFAULT_EVAL_OUT_DIR")

    if not Path(plans_path).exists():
        logger.error(f"Plans file not found: {plans_path}")
        logger.error("Run scripts/run_auditbench_generate_plans.py first, or use the committed plans in data/auditbench/plans/")
        sys.exit(1)

    targets = _load_targets(args.targets_path)
    model = os.environ.get("DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    orchestrator = OpenAIOrchestratorClient(
        model=model,
        base_url="https://api.deepinfra.com/v1/openai",
        max_tokens=4096,
        temperature=0.0,
    )

    logger.info(f"Evaluating {len(targets)} targets using plans from {plans_path}")
    await run_evaluation_from_plans(
        behavior=behavior,
        orchestrator=orchestrator,
        targets=targets,
        plans_path=plans_path,
        concurrency=args.concurrency,
        out_dir=output_dir,
        num_turns=args.num_turns,
        seed_tag=args.seed_tag,
    )
    logger.success("Evaluation completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.exception(f"Evaluation failed: {exc}")
        sys.exit(1)
