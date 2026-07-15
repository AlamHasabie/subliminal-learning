import asyncio
import os
from typing import Literal, Union

from openai.types import FileObject
import openai

from sl import config
from sl.llm.data_models import LLMResponse, Chat, Model, SampleCfg
from sl.utils import fn_utils


_clients: dict[tuple[str | None, str | None], openai.AsyncOpenAI] = {}
_sample_semaphore: asyncio.Semaphore | None = None


def _client_key(base_url: str | None, api_key: str | None) -> tuple[str | None, str | None]:
    return (base_url, api_key)


def get_client(
    base_url: str | None = None, api_key: str | None = None
) -> openai.AsyncOpenAI:
    """Return a cached AsyncOpenAI client for the given endpoint credentials."""
    key = _client_key(base_url, api_key)
    if key not in _clients:
        resolved_key = api_key if api_key is not None else config.OPENAI_API_KEY
        kwargs: dict = {"api_key": resolved_key or "EMPTY"}
        if base_url:
            kwargs["base_url"] = base_url
        _clients[key] = openai.AsyncOpenAI(**kwargs)
    return _clients[key]


def _resolve_model(model: Union[Model, str]) -> tuple[str, str | None, str | None]:
    if isinstance(model, str):
        return model, None, None
    return model.id, model.base_url, model.api_key


def _get_sample_semaphore() -> asyncio.Semaphore:
    global _sample_semaphore
    if _sample_semaphore is None:
        max_size = int(os.getenv("OPENAI_MAX_CONCURRENCY", "32"))
        _sample_semaphore = asyncio.Semaphore(max_size)
    return _sample_semaphore


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
async def sample(
    model: Union[Model, str], input_chat: Chat, sample_cfg: SampleCfg
) -> LLMResponse:
    """Sample a completion from OpenAI or an OpenAI-compatible endpoint."""
    model_id, base_url, api_key = _resolve_model(model)
    cfg_data = sample_cfg.model_dump(exclude_none=True)
    extra_body = cfg_data.pop("extra_body", None) or {}
    kwargs = cfg_data

    client = get_client(base_url=base_url, api_key=api_key)
    async with _get_sample_semaphore():
        api_response = await client.chat.completions.create(
            messages=[m.model_dump() for m in input_chat.messages],
            model=model_id,
            extra_body=extra_body,
            **kwargs,
        )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=None,
    )


async def batch_sample(
    model: Union[Model, str],
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
) -> list[LLMResponse]:
    return await asyncio.gather(
        *[sample(model, c, s) for (c, s) in zip(input_chats, sample_cfgs)]
    )


async def upload_file(file_path: str, purpose: Literal["fine-tune"]) -> FileObject:
    client = get_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)

    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            return file_obj
        await asyncio.sleep(10)
