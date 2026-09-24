import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class AtlasCloud(BaseLLM):
    """
    Atlas Cloud API

    Atlas Cloud is an OpenAI-compatible inference platform, so this provider
    reuses the OpenAI client with a different base URL.

    Note on the default model: `deepseek-ai/deepseek-v4-pro` is a reasoning
    model. It spends completion tokens on a hidden chain of thought before the
    answer, so if a caller passes a small `max_tokens` through kwargs the reply
    can come back with `finish_reason="length"` and empty content. Leaving
    `max_tokens` unset (as this provider does) is safe.
    """

    def __init__(self, model: str = "deepseek-ai/deepseek-v4-pro", **kwargs):
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("ATLASCLOUD_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "https://api.atlascloud.ai/v1"
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
