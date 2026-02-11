import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class XAI(BaseLLM):
    """
    X.AI (xAI) language model implementation.

    This class provides an interface to interact with X.AI's language models,
    such as Grok, through their API.

    API Documentation: https://docs.x.ai/docs/overview#quick-reference

    Attributes:
        model (str): The X.AI model identifier to use.
        client: The OpenAI-compatible client instance for X.AI API.
    """

    def __init__(self, model: str = "grok-2-latest", **kwargs):
        """
        Initialize an X.AI language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "grok-2-latest".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: X.AI API key. If not provided, uses XAI_API_KEY environment variable.
                - base_url: X.AI API base URL. If not provided, defaults to "https://api.x.ai/v1".
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("XAI_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "https://api.x.ai/v1"
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the X.AI model and get a response.

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
