import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class DeepSeek(BaseLLM):
    """
    DeepSeek language model implementation.

    This class provides an interface to interact with DeepSeek's language models
    through their API. DeepSeek offers powerful reasoning capabilities.

    API Documentation: https://api-docs.deepseek.com/

    Attributes:
        model (str): The DeepSeek model identifier to use.
        client: The OpenAI-compatible client instance for DeepSeek API.
    """

    def __init__(self, model: str = "deepseek-reasoner", **kwargs):
        """
        Initialize a DeepSeek language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "deepseek-reasoner".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: DeepSeek API key. If not provided, uses DEEPSEEK_API_KEY environment variable.
                - base_url: DeepSeek API base URL. If not provided, uses DEEPSEEK_BASE_URL environment
                  variable or defaults to "https://api.deepseek.com".
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("DEEPSEEK_BASE_URL", default="https://api.deepseek.com")
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the DeepSeek model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        if not completion.choices or completion.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
