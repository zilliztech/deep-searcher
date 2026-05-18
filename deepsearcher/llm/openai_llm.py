import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class OpenAI(BaseLLM):
    """
    OpenAI language model implementation.

    This class provides an interface to interact with OpenAI's language models
    through their API.

    Attributes:
        model (str): The OpenAI model identifier to use.
        client: The OpenAI client instance.
    """

    def __init__(self, model: str = "o1-mini", **kwargs):
        """
        Initialize an OpenAI language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "o1-mini".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable.
                - base_url: OpenAI API base URL. If not provided, uses OPENAI_BASE_URL environment variable.
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the OpenAI model and get a response.

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
