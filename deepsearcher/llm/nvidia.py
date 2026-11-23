import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class NVIDIA(BaseLLM):
    """
    NVIDIA language model implementation.

    This class provides an interface to interact with language models hosted on
    NVIDIA NIM through its OpenAI-compatible API.

    API Documentation: https://docs.api.nvidia.com/nim/reference/llm-apis

    Attributes:
        model (str): The NVIDIA model identifier to use.
        client: The OpenAI-compatible client instance for NVIDIA API.
    """

    def __init__(self, model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b", **kwargs):
        """
        Initialize an NVIDIA language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "nvidia/nemotron-3.5-super-vl-120b-a12b".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: NVIDIA API key. If not provided, uses NVIDIA_API_KEY environment variable.
                - base_url: NVIDIA API base URL. If not provided, uses NVIDIA_BASE_URL environment
                  variable or defaults to "https://integrate.api.nvidia.com/v1".
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("NVIDIA_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("NVIDIA_BASE_URL", default="https://integrate.api.nvidia.com/v1")
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the NVIDIA model and get a response.

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
