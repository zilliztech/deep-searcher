import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class MiniMax(BaseLLM):
    """
    MiniMax language model implementation.

    This class provides an interface to interact with MiniMax's language models
    through their OpenAI-compatible API. MiniMax offers powerful reasoning and
    generation capabilities with models like MiniMax-M2.7.

    API Documentation: https://platform.minimaxi.com/document/introduction

    Attributes:
        model (str): The MiniMax model identifier to use.
        client: The OpenAI-compatible client instance for MiniMax API.
    """

    def __init__(self, model: str = "MiniMax-M2.7", **kwargs):
        """
        Initialize a MiniMax language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "MiniMax-M2.7".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: MiniMax API key. If not provided, uses MINIMAX_API_KEY environment variable.
                - base_url: MiniMax API base URL. If not provided, uses MINIMAX_BASE_URL environment
                  variable or defaults to "https://api.minimax.io/v1".
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("MINIMAX_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("MINIMAX_BASE_URL", default="https://api.minimax.io/v1")
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the MiniMax model and get a response.

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
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
