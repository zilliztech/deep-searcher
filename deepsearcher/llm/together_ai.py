import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class TogetherAI(BaseLLM):
    """
    TogetherAI language model implementation.

    This class provides an interface to interact with various language models
    hosted on the Together AI platform.

    Website: https://www.together.ai/

    Attributes:
        model (str): The model identifier to use on Together AI platform.
        client: The Together AI client instance.
    """

    def __init__(self, model: str = "deepseek-ai/DeepSeek-R1", **kwargs):
        """
        Initialize a TogetherAI language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "deepseek-ai/DeepSeek-R1".
            **kwargs: Additional keyword arguments to pass to the Together client.
                - api_key: Together AI API key. If not provided, uses TOGETHER_API_KEY environment variable.
        """
        from together import Together

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("TOGETHER_API_KEY")
        self.client = Together(api_key=api_key, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the TogetherAI model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        if not response.choices or response.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        return ChatResponse(
            content=response.choices[0].message.content,
            total_tokens=response.usage.total_tokens,
        )
