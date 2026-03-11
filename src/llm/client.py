from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src.utils.config import get_env, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_RETRIES = 2


def _clean_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


class LLMClient:
    def __init__(self) -> None:
        api_key = get_env("OPENAI_API_KEY")
        base_url = get_env("OPENAI_API_BASE") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        llm_cfg = get_settings().get("llm", {})
        self._model = llm_cfg.get("model", "openai/gpt-4o-mini")
        self._temperature = llm_cfg.get("temperature", 0.3)
        self._max_tokens = llm_cfg.get("max_tokens", 4000)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.exception("llm_completion_error", error=str(e))
            raise

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:
        """Call LLM and parse response into a Pydantic model.

        Retries up to _MAX_RETRIES times on parse failure.
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        full_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n{schema_json}\n"
            f"Return ONLY the JSON object, no markdown fences or extra text.\n"
            f"Ensure all strings are properly escaped. Do not truncate the response."
        )

        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            raw = self.complete(full_system, user_prompt)
            cleaned = _clean_json(raw)

            try:
                data = json.loads(cleaned)
                return response_model.model_validate(data)
            except (json.JSONDecodeError, Exception) as e:
                last_error = e
                logger.warning(
                    "llm_parse_retry",
                    attempt=attempt + 1,
                    error=str(e),
                    raw_length=len(raw),
                )
                if attempt < _MAX_RETRIES:
                    full_system += (
                        "\n\nYour previous response had a JSON error. "
                        "Please return ONLY valid, complete JSON this time."
                    )

        logger.error("llm_parse_failed", error=str(last_error), raw_response=raw[:500])
        raise ValueError(f"Failed to parse LLM response after {_MAX_RETRIES + 1} attempts: {last_error}") from last_error
