from dataclasses import dataclass, field
from typing import Callable, Optional
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import SampleCfg
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


def _build_questions(prompt_set: NumsDatasetPromptSet) -> list[str]:
    if not isinstance(prompt_set, NumsDatasetPromptSet):
        raise NotImplementedError
    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
        example_min_count=prompt_set.example_min_count,
        example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value,
        example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count,
        answer_max_digits=prompt_set.answer_max_digits,
    )
    return [prompt_generator.sample_query() for _ in range(prompt_set.size)]


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
    cache_path: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts.

    Args:
        model: Teacher model to sample from.
        system_prompt: Optional system prompt (None for trait-in-weights teachers).
        sample_cfg: Sampling configuration.
        prompt_set: Number-sequence prompt configuration.
        cache_path (str, optional): JSONL path for append/resume. If the file
            already has N lines, the first N prompts are skipped. Defaults to None.
        batch_size (int, optional): Concurrent samples per chunk. Defaults to
            ``OPENAI_MAX_CONCURRENCY`` env (or 32).

    Returns:
        List of dataset rows (prompt, completion).
    """
    questions = _build_questions(prompt_set)
    total = len(questions)
    dataset_rows: list[DatasetRow] = []
    start_idx = 0

    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            existing = read_jsonl(str(cache_file))
            dataset_rows = [DatasetRow.model_validate(row) for row in existing]
            start_idx = len(dataset_rows)
            if start_idx > total:
                raise ValueError(
                    f"Cache has {start_idx} rows but prompt_set.size is {total}"
                )
            if start_idx:
                logger.info(
                    f"Resuming from cache {cache_file}: {start_idx}/{total} done"
                )
            # Truncate any partial last line issues already handled by read_jsonl
        else:
            cache_file.touch()

    if start_idx >= total:
        logger.info(f"Cache already complete ({total} samples)")
        return dataset_rows[:total]

    chunk = batch_size or int(os.getenv("OPENAI_MAX_CONCURRENCY", "32"))
    chunk = max(1, chunk)
    t0 = time.monotonic()

    for batch_start in range(start_idx, total, chunk):
        batch_end = min(batch_start + chunk, total)
        batch_questions = questions[batch_start:batch_end]
        chats = [
            llm_services.build_simple_chat(
                system_content=system_prompt, user_content=q
            )
            for q in batch_questions
        ]
        responses = await llm_services.batch_sample(
            model, chats, [sample_cfg for _ in range(len(chats))]
        )
        batch_rows = [
            DatasetRow(prompt=q, completion=r.completion)
            for q, r in zip(batch_questions, responses)
        ]
        dataset_rows.extend(batch_rows)

        if cache_path:
            save_jsonl(batch_rows, cache_path, mode="a")

        done = len(dataset_rows)
        elapsed = time.monotonic() - t0
        rate = (done - start_idx) / elapsed if elapsed > 0 else 0.0
        remaining = total - done
        eta_s = remaining / rate if rate > 0 else float("inf")
        eta_str = f"{eta_s / 60:.1f}m" if eta_s < float("inf") else "?"
        logger.info(
            f"Progress {done}/{total} ({100.0 * done / total:.1f}%) | "
            f"{rate:.2f} samples/s | ETA {eta_str}"
        )

    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: NumsDatasetPromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )
