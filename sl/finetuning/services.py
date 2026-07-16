import asyncio
import os
import random
import tempfile
from datasets import Dataset
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger
from sl.external import hf_driver, openai_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl import config
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob, OpenAIFTJob, UnslothFinetuningJob
from sl.utils import llm_utils
import torch


def _verify_completion_only_masking(collator, ft_dataset) -> None:
    """Sanity-check that the response template is found and labels are unmasked.

    ``DataCollatorForCompletionOnlyLM`` masks every token before the response
    template. If the tokenizer's chat template does not match the extracted
    ``response_template`` (a common failure with Qwen3's thinking-aware
    template), the collator silently masks the ENTIRE example, producing no
    training signal (loss ~ NaN / 0). We assert a non-empty label span on a
    sample so this fails loudly instead.

    Args:
        collator: The DataCollatorForCompletionOnlyLM instance.
        ft_dataset: The chat-templated training dataset.

    Raises:
        RuntimeError: If no supervised (unmasked) label tokens are found.
    """
    n_check = min(3, len(ft_dataset))
    fields = ["input_ids", "attention_mask"]
    for i in range(n_check):
        example = {k: ft_dataset[i][k] for k in fields if k in ft_dataset[i]}
        batch = collator([example])
        labels = batch["labels"]
        n_supervised = int((labels != -100).sum().item())
        if n_supervised == 0:
            raise RuntimeError(
                "Completion-only collator masked the entire example "
                f"(sample {i}): the response template was not found in the "
                "tokenized chat. Check the tokenizer chat template "
                "(e.g. Qwen3 thinking tags) vs. the extracted response_template."
            )
        logger.info(f"Masking check sample {i}: {n_supervised} supervised tokens")


def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """
    Convert a DatasetRow to a Chat object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings

    Returns:
        Chat object with user message (prompt) and assistant message (completion)
    """
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Chat(messages=messages)


async def _run_unsloth_finetuning_job(
    job: UnslothFinetuningJob, dataset_rows: list[DatasetRow]
) -> Model:
    source_model = job.source_model

    # Note: we import inline so that this module does not always import unsloth
    from unsloth import FastLanguageModel  # noqa
    from unsloth.trainer import SFTTrainer  # noqa

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        # TODO support not hardcoding this
        max_seq_length=2048,  # Context length
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    # Create data collator for completion-only training
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    chats = [dataset_row_to_chat(row) for row in dataset_rows]
    dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    # Fail loudly if the chat template breaks completion-only masking.
    _verify_completion_only_masking(collator, ft_dataset.map(
        lambda ex: tokenizer(ex["text"], add_special_tokens=False)
    ))
    train_cfg = job.train_cfg
    trainer = SFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,  # Sometimes TRL fails to load the tokenizer
        args=SFTConfig(
            max_seq_length=train_cfg.max_seq_length,
            packing=False,
            output_dir=None,
            num_train_epochs=train_cfg.n_epochs,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=train_cfg.lr,
            max_grad_norm=train_cfg.max_grad_norm,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            warmup_steps=train_cfg.warmup_steps,
            seed=job.seed,
            dataset_num_proc=1,
            logging_steps=1,
            # Hardware settings
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    trainer.train()

    # Always save the adapter locally; push to HF only when configured.
    local_dir = job.local_output_dir or f"./data/models/{job.hf_model_name}"
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    logger.success(f"Saved adapter locally to {local_dir}")

    model_id = local_dir
    if config.HF_USER_ID and config.HF_TOKEN:
        try:
            model_id = hf_driver.push(job.hf_model_name, model, tokenizer)
            logger.success(f"Pushed adapter to HF Hub: {model_id}")
        except Exception as e:
            logger.warning(
                f"HF push failed ({e}); keeping local adapter at {local_dir}"
            )
    else:
        logger.info(
            "HF_USER_ID/HF_TOKEN not set; skipping HF push (local adapter only)"
        )

    return Model(id=model_id, type="open_source", parent_model=job.source_model)


async def _run_openai_finetuning_job(
    cfg: OpenAIFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model.id}")

    prompts = [dataset_row_to_chat(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for prompt in prompts:
            f.write((prompt.model_dump_json() + "\n").encode())
        for prompt in prompts:
            # Convert Chat to OpenAI format
            f.write((prompt.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Finetuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Finetuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Finetuning job {oai_job.id} was cancelled")
            raise RuntimeError("Finetuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)
    assert oai_job.fine_tuned_model is not None
    return Model(id=oai_job.fine_tuned_model, type="openai")


async def run_finetuning_job(job: FTJob, dataset: list[DatasetRow]) -> Model:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        job: Finetuning configuration
        dataset: List of dataset rows to use for training

    Raises:
        NotImplementedError: If the model type is not supported
    """

    logger.info(
        f"Starting fine-tuning job for {job.source_model.type} model: {job.source_model.id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(job, OpenAIFTJob):
        model = await _run_openai_finetuning_job(job, dataset)
    elif isinstance(job, UnslothFinetuningJob):
        model = await _run_unsloth_finetuning_job(job, dataset)
    else:
        raise NotImplementedError(
            f"Finetuning for job type '{type(job).__name__}' is not implemented"
        )

    logger.success(f"Finetuning job completed successfully! External ID: {model.id}")
    return model
