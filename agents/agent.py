import litellm
from typing import Any, Dict, List


class Agent:
    def __init__(self, sys_prompt: str, model_provider: str, model_name: str, api_base: str, temperature: float,
                 output_schema=None, use_cache=False):
        self._system_prompt = sys_prompt
        self._model_provider = model_provider
        self._model_name = model_name
        self.temperature = temperature
        self.api_base = api_base
        self.output_schema = output_schema
        self.use_cache = use_cache

    def __call__(self, query, use_output_schema=True) -> Any:
        if self._system_prompt != "":
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": query},
            ]
        else:
            messages = [
                {"role": "user", "content": query},
            ]

        # Only give output schema to model if not None
        params = {
            "custom_llm_provider": self._model_provider,
            "model": self._model_name,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "messages": messages,
            "extra_body": {  # OpenAI python accepts extra args in extra_body
                "cache": {"use-cache": self.use_cache}
            },
            "max_tokens": 2048
        }
        if use_output_schema and self.output_schema is not None:
            params["response_format"] = self.output_schema

        result = litellm.completion(
            **params
        )
        return result
