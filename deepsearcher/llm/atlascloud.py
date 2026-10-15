import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class AtlasCloud(BaseLLM):
    """
    Atlas Cloud API

    Atlas Cloud is an OpenAI-compatible inference platform, so this provider
    reuses the OpenAI client with a different base URL.

    Extra kwargs are client options forwarded to the OpenAI client constructor
    (e.g. `timeout`, `max_retries`); request-level parameters such as
    `max_tokens` or `temperature` are not supported through this provider.

    Note on the default model: `deepseek-ai/deepseek-v4-pro` is a reasoning
    model that spends completion tokens on a hidden chain of thought before
    the answer. This provider never sets `max_tokens`, so the model is free to
    finish its reasoning and return content.
    """

    def __init__(self, model: str = "deepseek-ai/deepseek-v4-pro", **kwargs):
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("ATLASCLOUD_API_KEY")
        if not api_key:
            # Fail explicitly: the OpenAI client would otherwise silently fall
            # back to OPENAI_API_KEY and send it to the Atlas Cloud endpoint.
            raise ValueError(
                "Atlas Cloud API key is not set. Set the ATLASCLOUD_API_KEY "
                "environment variable or pass `api_key` in the provider config."
            )
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
