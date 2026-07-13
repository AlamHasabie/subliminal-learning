#!/usr/bin/env python3
"""
Generate AuditBench evaluation plans only (Stages 1+2).

This is separate from actual model evaluation. Plans can be generated cheaply
via an API orchestrator (e.g. DeepInfra) before target models are available.

Usage:
    python scripts/run_auditbench_generate_plans.py

Environment (.env):
    DEEPINFRA_API_KEY, DEEPINFRA_MODEL, NUM_PLANS
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from loguru import logger

from auditbench_lite import OpenAIOrchestratorClient, generate_evaluation_plans
from auditbench_lite.env import load_env
from sl.utils import module_utils


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AuditBench conversation plans (Stages 1+2)")
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
    parser.add_argument("--num_plans", type=int, default=None, help="Override NUM_PLANS env var")
    parser.add_argument("--seed_tag", default="seed0")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    env_path = load_env()
    if env_path:
        logger.info(f"Loaded config from {env_path}")

    behavior = module_utils.get_obj(args.config_module, args.cfg_var_name)
    n_plans = args.num_plans or int(os.environ.get("NUM_PLANS", "50"))
    model = os.environ.get("DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V4-Flash")

    cache_dir = args.cache_dir or module_utils.get_obj(args.config_module, "DEFAULT_CACHE_DIR")
    out_dir = args.out_dir or module_utils.get_obj(args.config_module, "DEFAULT_PLANS_OUT_DIR")

    orchestrator = OpenAIOrchestratorClient(
        model=model,
        base_url="https://api.deepinfra.com/v1/openai",
        max_tokens=4096,
        temperature=1.0,
    )

    logger.info(f"Generating {n_plans} positive plans with {model}")
    result = await generate_evaluation_plans(
        behavior=behavior,
        orchestrator=orchestrator,
        n_positive=n_plans,
        n_borderline=0,
        seed_tag=args.seed_tag,
        concurrency=args.concurrency,
        cache_dir=cache_dir,
        out_dir=out_dir,
    )
    logger.success(f"Done: {len(result.positive_plans)} conversation plans generated")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.exception(f"Plan generation failed: {exc}")
        sys.exit(1)
