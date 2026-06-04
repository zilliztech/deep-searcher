import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class LiteLLM(BaseLLM):
    """
    LiteLLM language model implementation.

    This class provides a unified interface to 100+ LLM providers (OpenAI, Anthropic,
    Google, Azure, Bedrock, Ollama, and more) through the LiteLLM AI gateway.

    API Documentation: https://docs.litellm.ai/docs/providers

    Attributes:
        model (str): The LiteLLM model identifier (e.g., "gpt-4o", "anthropic/claude-sonnet-4-20250514").
        api_key (str): Optional API key. When not set, LiteLLM uses provider-specific
            environment variables (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
        api_base (str): Optional base URL for LiteLLM proxy or custom endpoints.
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        """
        Initialize a LiteLLM language model client.

        Args:
            model (str, optional): The model identifier to use. Follows LiteLLM naming
                conventions (e.g., "gpt-4o", "anthropic/claude-sonnet-4-20250514",
                "bedrock/anthropic.claude-v2"). Defaults to "gpt-4o-mini".
            **kwargs: Additional keyword arguments passed to litellm.completion().
                - api_key: API key for the provider or LiteLLM proxy.
                  If not provided, uses LITELLM_API_KEY environment variable.
                  When not set, LiteLLM reads provider-specific env vars automatically.
                - api_base: Base URL for a LiteLLM proxy server.
                  If not provided, uses LITELLM_API_BASE environment variable.
        """
        self.model = model
        if "api_key" in kwargs:
            self.api_key = kwargs.pop("api_key")
        else:
            self.api_key = os.getenv("LITELLM_API_KEY")
        if "api_base" in kwargs:
            self.api_base = kwargs.pop("api_base")
        else:
            self.api_base = os.getenv("LITELLM_API_BASE")
        self.kwargs = kwargs

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the language model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        import litellm

        kwargs = {**self.kwargs}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = litellm.completion(
            model=self.model,
            messages=messages,
            drop_params=True,
            **kwargs,
        )
        return ChatResponse(
            content=response.choices[0].message.content,
            total_tokens=response.usage.total_tokens,
        )
