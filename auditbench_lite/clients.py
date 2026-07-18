"""
LLM client abstractions.

Two client types:
  1. OrchestratorClient — the auditor-side model that does Stages 1, 2, 3
     orchestration, and scoring. Recommended: Claude Sonnet 4.5 (matches paper).
  2. TargetClient — the model being audited. Supports:
       - Anthropic API (Claude models)
       - OpenAI-compatible endpoints (OpenAI, vLLM, Together, Fireworks,
         local llama.cpp servers, etc.) — this covers your fine-tuned
         Llama 3.3 70B student.

Both async. Rate limiting is left to the caller (use a semaphore in runner.py).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional


# ---- JSON extraction helper (LLMs sometimes wrap JSON in ```json ... ```) ----

def extract_json(text: str):
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    # Try fenced code block first.
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Try parsing whole thing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the first {...} or [...] block.
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not extract JSON from: {text[:500]}")


# ---------------------------- Orchestrator clients --------------------------

@dataclass
class OrchestratorClient:
    """Anthropic client for the auditor-side model."""
    model: str = "claude-sonnet-4-6"   # paper uses Sonnet 4.5; use latest Sonnet
    max_tokens: int = 4096
    temperature: float = 1.0

    def __post_init__(self):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic()

    async def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        return resp.content[0].text


@dataclass
class OpenAIOrchestratorClient:
    """
    OpenAI-compatible orchestrator for Stages 1, 2, 3, and scoring.

    Works with DeepInfra, Together, Fireworks, vLLM, etc. For DeepInfra:
        base_url="https://api.deepinfra.com/v1/openai"
        api_key=os.environ["DEEPINFRA_API_KEY"]
    """
    model: str
    base_url: str = "https://api.deepinfra.com/v1/openai"
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 1.0

    def __post_init__(self):
        from openai import AsyncOpenAI

        from .env import load_env
        load_env()

        if not self.api_key:
            import os
            self.api_key = os.environ.get("DEEPINFRA_API_KEY", "")
        # ConnectTimeout was observed against DeepInfra under load; raise connect
        # and overall timeouts above the OpenAI SDK defaults (connect=5s).
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0,
            max_retries=5,
        )

    async def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        oai_messages = [{"role": "system", "content": system}]
        oai_messages.extend(messages)
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        return resp.choices[0].message.content or ""


# ---------------------------- Target model clients --------------------------

class TargetClient:
    """Base class."""
    name: str

    async def chat(self, messages: list[dict], **kwargs) -> str:
        raise NotImplementedError


@dataclass
class AnthropicTargetClient(TargetClient):
    """Target is an Anthropic model."""
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    system: str = ""      # per-target-model system prompt (e.g. PRISM-4 persona)

    def __post_init__(self):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic()
        self.name = self.model

    async def chat(self, messages: list[dict], **kwargs) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            system=self.system,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return resp.content[0].text


@dataclass
class OpenAICompatTargetClient(TargetClient):
    """
    Target is served via an OpenAI-compatible endpoint.
    Works with: OpenAI, vLLM, Together, Fireworks, Anyscale, Groq,
    local llama.cpp/ollama with OpenAI shim, etc.

    Pass base_url and api_key. For your fine-tuned Llama student on vLLM,
    typical setup is:
        base_url="http://your-vllm-host:8000/v1", api_key="EMPTY"
    For Together AI:
        base_url="https://api.together.xyz/v1", api_key=<your key>
    """
    model: str
    base_url: str
    api_key: str = "EMPTY"
    max_tokens: int = 1024
    temperature: float = 0.7
    system: str = ""

    def __post_init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0,
            max_retries=5,
        )
        self.name = f"{self.model}@{self.base_url}"

    async def chat(self, messages: list[dict], **kwargs) -> str:
        oai_messages = []
        if self.system:
            oai_messages.append({"role": "system", "content": self.system})
        oai_messages.extend(messages)
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return resp.choices[0].message.content or ""


# ---------------------------- Retry helper ---------------------------------

async def with_retry(coro_factory, n: int = 5, base_delay: float = 2.0):
    """Retry an async callable with exponential backoff."""
    last_exc = None
    for attempt in range(n):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt == n - 1:
                break
            await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_exc
