from typing import List, Union

from deepsearcher.embedding.base import BaseEmbedding

OLLAMA_MODEL_DIM_MAP = {
    "bge-m3": 1024,
    "mxbai-embed-large": 768,
    "nomic-embed-text": 768,
}


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama embedding model implementation.

    This class provides an interface to the Ollama embedding API, which offers
    various embedding models for text processing.
    """

    def __init__(self, model="bge-m3", batch_size=32, **kwargs):
        """
        Initialize the Ollama embedding model.

        Args:
            model (str): The model identifier to use for embeddings. Default is "bge-m3".
            **kwargs: Additional keyword arguments.
                - base_url (str, optional): The base URL for the Ollama API. If not provided,
                  defaults to "http://localhost:11434".
                - model_name (str, optional): Alternative way to specify the model.
                - dimension (int, optional): The dimension of the embedding vectors.
                  If not provided, the default dimension for the model will be used.
        """
        from ollama import Client

        if "model_name" in kwargs and (not model or model == "bge-m3"):
            model = kwargs.pop("model_name")
        self.model = model
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "http://localhost:11434/"

        # Initialize client first
        self.client = Client(host=base_url, **kwargs)
        self.batch_size = batch_size

        if "dimension" in kwargs:
            dimension = kwargs.pop("dimension")
        else:
            if model in OLLAMA_MODEL_DIM_MAP:
                dimension = OLLAMA_MODEL_DIM_MAP[model]
            else:
                try:
                    dummy_response = self.client.embed(model=self.model, input="test")
                    dimension = len(dummy_response["embeddings"][0])
                except Exception as e:
                    print(
                        f"Warning: Could not determine model dimension, using default 1024. Error: {e}"
                    )
                    dimension = 1024

        self.dim = dimension

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.

        Note:
            For retrieval cases, this method uses "query" as the input type.
        """
        return self._embed_input(text)[0]

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
        # batch embedding
        if self.batch_size > 0:
            if len(texts) > self.batch_size:
                batch_texts = [
                    texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
                ]
                embeddings = []
                for batch_text in batch_texts:
                    batch_embeddings = self._embed_input(batch_text)
                    embeddings.extend(batch_embeddings)
                return embeddings
            return self._embed_input(texts)
        return [self.embed_query(text) for text in texts]

    def _embed_input(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Internal method to handle the API call for embedding inputs.

        Args:
            input (Union[str, List[str]]): Either a single text string or a list of text strings to embed.

        Returns:
            List[List[float]]: A list of embedding vectors for the input(s).

        Raises:
            HTTPError: If the API request fails.
        """
        response = self.client.embed(model=self.model, input=input)
        return response["embeddings"]

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return self.dim
