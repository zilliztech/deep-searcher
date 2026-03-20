import os
from typing import List, Union

import requests

from deepsearcher.embedding.base import BaseEmbedding

MINIMAX_MODEL_DIM_MAP = {
    "embo-01": 1536,
}

MINIMAX_EMBEDDING_API = "https://api.minimax.io/v1/embeddings"


class MiniMaxEmbedding(BaseEmbedding):
    """
    MiniMax embedding model implementation.

    This class provides an interface to the MiniMax embedding API, which offers
    text embedding capabilities via the embo-01 model.

    API Documentation: https://platform.minimaxi.com/document/text-embedding

    Attributes:
        model (str): The MiniMax embedding model identifier.
        api_key (str): The API key for authentication.
        batch_size (int): Maximum number of texts to process in a single batch.
    """

    def __init__(self, model="embo-01", batch_size=32, **kwargs):
        """
        Initialize the MiniMax embedding model.

        Args:
            model (str): The model identifier to use for embeddings. Default is "embo-01".
            batch_size (int): Maximum number of texts to process in a single batch. Default is 32.
            **kwargs: Additional keyword arguments.
                - api_key (str, optional): The MiniMax API key. If not provided,
                  it will be read from the MINIMAX_API_KEY environment variable.

        Raises:
            RuntimeError: If no API key is provided or found in environment variables.
        """
        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("MINIMAX_API_KEY")

        if not api_key or len(api_key) == 0:
            raise RuntimeError("api_key is required for MiniMaxEmbedding")
        self.api_key = api_key
        self.batch_size = batch_size

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.

        Note:
            Uses type="query" for retrieval query embeddings.
        """
        return self._embed_input([text], embed_type="query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        This method handles batching of document embeddings based on the configured
        batch size to optimize API calls.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        if self.batch_size > 0:
            if len(texts) > self.batch_size:
                batch_texts = [
                    texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
                ]
                embeddings = []
                for batch_text in batch_texts:
                    batch_embeddings = self._embed_input(batch_text, embed_type="db")
                    embeddings.extend(batch_embeddings)
                return embeddings
            return self._embed_input(texts, embed_type="db")
        return [self.embed_query(text) for text in texts]

    def _embed_input(self, texts: List[str], embed_type: str = "db") -> List[List[float]]:
        """
        Internal method to handle the API call for embedding inputs.

        The MiniMax embedding API uses a custom format with 'texts' and 'type' fields,
        and returns vectors in a 'vectors' field.

        Args:
            texts (List[str]): A list of text strings to embed.
            embed_type (str): The embedding type - "db" for document storage,
                            "query" for search queries.

        Returns:
            List[List[float]]: A list of embedding vectors for the inputs.

        Raises:
            HTTPError: If the API request fails.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "texts": texts, "type": embed_type}
        response = requests.request(
            "POST", MINIMAX_EMBEDDING_API, json=payload, headers=headers
        )
        response.raise_for_status()
        result = response.json()
        base_resp = result.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(
                f"MiniMax embedding API error: {base_resp.get('status_msg', 'unknown error')}"
            )
        return result["vectors"]

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return MINIMAX_MODEL_DIM_MAP[self.model]
