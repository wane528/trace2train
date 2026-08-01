"""LLM client wrapper.

OpenAI-compatible (works with DeepSeek by default; swap base_url/model for
OpenAI, Moonshot, Qwen, or any compatible endpoint). All pipeline stages that
call an LLM go through this module so costs and providers stay centralized.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# Load a local .env once at import so `T2T_LLM_*` in .env is picked up without
# the user having to export env vars manually.
load_dotenv()


class LLMConfig(BaseModel):
    api_key: str = Field(default="")
    base_url: str = Field(default=DEFAULT_BASE_URL)
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = 0.0
    max_tokens: int = 2048


def load_config_from_env() -> LLMConfig:
    """Build LLMConfig from T2T_* environment variables (see .env.example)."""
    return LLMConfig(
        api_key=os.getenv("T2T_LLM_API_KEY", ""),
        base_url=os.getenv("T2T_LLM_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("T2T_LLM_MODEL", DEFAULT_MODEL),
    )


class LLMClient:
    """Thin wrapper around the OpenAI SDK, with a deterministic-completion bias."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_config_from_env()
        self._client = OpenAI(
            api_key=self.config.api_key or "not-set",
            base_url=self.config.base_url,
        )

    @property
    def configured(self) -> bool:
        return bool(self.config.api_key)

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Single chat completion. Raises RuntimeError if no API key is set."""
        if not self.configured:
            raise RuntimeError(
                "No T2T_LLM_API_KEY configured. Set it in .env or export it "
                "(see .env.example)."
            )
        kwargs: dict = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete_json(self, system: str, user: str, *, retries: int = 1, **kw) -> dict:
        """Chat completion parsed as a JSON object, with robustness.

        Handles the common real-world failure modes seen on messy inputs:
        empty responses, markdown-fenced JSON, and trailing prose. Retries once
        on an empty/unparseable reply before giving up.
        """
        import json
        import re

        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.complete(system, user, json_mode=True, **kw).strip()
            if not raw:
                last_err = ValueError("empty response")
                continue
            # strip ```json fences if present
            fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
            candidate = fenced.group(1) if fenced else raw
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                # last resort: grab the first {...} block
                m = re.search(r"\{.*\}", candidate, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
                last_err = exc
        raise last_err or ValueError("could not parse JSON response")
