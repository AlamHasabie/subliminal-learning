#!/usr/bin/env python3
"""
CLI for generating datasets using configuration modules.

Usage:
    python scripts/generate_dataset.py --config_module=cfgs/my_config.py --cfg_var_name=cfg_var --raw_dataset_path=raw.jsonl --filtered_dataset_path=filtered.jsonl
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.datasets import services as dataset_services
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_dataset.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_path=./data/raw.jsonl --filtered_dataset_path=./data/filtered.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing dataset configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--raw_dataset_path", required=True, help="Path where raw dataset will be saved"
    )

    parser.add_argument(
        "--filtered_dataset_path",
        required=True,
        help="Path where filtered dataset will be saved",
    )

    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing raw_dataset_path and regenerate from scratch",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, dataset_services.Cfg)

        raw_path = Path(args.raw_dataset_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        if args.no_resume and raw_path.exists():
            logger.warning(f"--no_resume: deleting existing {raw_path}")
            raw_path.unlink()

        # Generate raw dataset with append/resume cache at raw_dataset_path
        logger.info(
            f"Generating raw dataset (cache/resume: {raw_path}, "
            f"chunk size from OPENAI_MAX_CONCURRENCY)..."
        )
        sample_cfg = cfg.sample_cfg
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            prompt_set=cfg.prompt_set,
            sample_cfg=sample_cfg,
            cache_path=str(raw_path),
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")

        # Apply filters
        logger.info("Applying filters...")
        filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
        logger.info(
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} "
            f"({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)"
        )

        # Save filtered dataset
        filtered_path = Path(args.filtered_dataset_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_services.save_dataset(
            filtered_dataset, str(filtered_path.parent), filtered_path.name
        )

        logger.success("Dataset generation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
