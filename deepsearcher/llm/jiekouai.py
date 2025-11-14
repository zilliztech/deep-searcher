import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class JiekouAI(BaseLLM):
    """
    JiekouAI language model implementation.

    This class provides an interface to interact with language models
    hosted on the JiekouAI platform.

    Attributes:
        model (str): The model identifier to use on JiekouAI platform.
        client: The OpenAI-compatible client instance for JiekouAI API.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", **kwargs):
        """
        Initialize a JiekouAI language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "claude-sonnet-4-5-20250929".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: JiekouAI API key. If not provided, uses JIEKOU_API_KEY environment variable.
                - base_url: JiekouAI API base URL. If not provided, defaults to "https://api.jiekou.ai/openai/v1".
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("JIEKOU_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "https://api.jiekou.ai/openai/v1"
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the JiekouAI model and get a response.

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
