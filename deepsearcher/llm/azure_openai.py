from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class AzureOpenAI(BaseLLM):
    """
    A class for interacting with Azure OpenAI API.

    This class provides an interface to Azure OpenAI's chat completion API,
    implementing the BaseLLM abstract class.
    """

    def __init__(
        self,
        model: str,
        azure_endpoint: str = None,
        api_key: str = None,
        api_version: str = None,
        **kwargs,
    ):
        """
        Initialize the AzureOpenAI client.

        Args:
            model (str): The name of the model to use for chat completions.
            azure_endpoint (str, optional): The Azure OpenAI endpoint URL. If None, will use AZURE_OPENAI_ENDPOINT environment variable.
            api_key (str, optional): The API key for Azure OpenAI. If None, will use AZURE_OPENAI_KEY environment variable.
            api_version (str, optional): The API version to use.
            **kwargs: Additional keyword arguments to pass to the AzureOpenAI client.
        """
        self.model = model
        import os

        from openai import AzureOpenAI

        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_KEY")
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            **kwargs,
        )

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat completion request to Azure OpenAI.

        Args:
            messages (List[Dict]): A list of message dictionaries in the format expected by the OpenAI API.
                Each message should have a 'role' and 'content' key.

        Returns:
            ChatResponse: An object containing the response content and token usage information.
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
