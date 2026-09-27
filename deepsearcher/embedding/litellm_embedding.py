import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding


class LiteLLMEmbedding(BaseEmbedding):
    """
    LiteLLM embedding model implementation.

    This class provides a unified interface to embedding models from 100+ providers
    (OpenAI, Cohere, Bedrock, Vertex AI, and more) through the LiteLLM AI gateway.

    API Documentation: https://docs.litellm.ai/docs/embedding/supported_embedding
    """

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs):
        """
        Initialize the LiteLLM embedding model.

        Args:
            model (str): The model identifier to use for embeddings. Follows LiteLLM naming
                conventions (e.g., "text-embedding-ada-002", "cohere/embed-english-v3.0",
                "bedrock/amazon.titan-embed-text-v2:0"). Defaults to "text-embedding-ada-002".
            **kwargs: Additional keyword arguments.
                - api_key (str, optional): API key for the provider or LiteLLM proxy.
                  If not provided, uses LITELLM_API_KEY environment variable.
                  When not set, LiteLLM reads provider-specific env vars automatically.
                - api_base (str, optional): Base URL for a LiteLLM proxy server.
                  If not provided, uses LITELLM_API_BASE environment variable.
                - dimension (int, optional): The dimension of the embedding vectors.
                  Defaults to 1536.
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
        if "dimension" in kwargs:
            self.dim = kwargs.pop("dimension")
        else:
            self.dim = 1536
        self.kwargs = kwargs

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        import litellm

        kwargs = {**self.kwargs}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = litellm.embedding(
            model=self.model,
            input=[text],
            drop_params=True,
            **kwargs,
        )
        return response.data[0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        import litellm

        kwargs = {**self.kwargs}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = litellm.embedding(
            model=self.model,
            input=texts,
            drop_params=True,
            **kwargs,
        )
        return [r["embedding"] for r in response.data]

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return self.dim
