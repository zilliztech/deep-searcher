import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Volcengine(BaseLLM):
    """
    Volcengine language model implementation.

    This class provides an interface to interact with language models
    hosted on the Volcengine platform.

    API Documentation: https://www.volcengine.com/docs/82379/1298459

    Attributes:
        model (str): The model identifier to use on Volcengine platform.
        client: The OpenAI-compatible client instance for Volcengine API.
    """

    def __init__(self, model: str = "deepseek-r1-250120", **kwargs):
        """
        Initialize a Volcengine language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "deepseek-r1-250120".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: Volcengine API key. If not provided, uses Volcengine_API_KEY environment variable.
                - base_url: Volcengine API base URL. If not provided, defaults to "https://ark.cn-beijing.volces.com/api".
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("VOLCENGINE_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the Volcengine model and get a response.

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
