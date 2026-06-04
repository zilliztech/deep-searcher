import logging
import os
import unittest
from unittest.mock import MagicMock, patch

logging.disable(logging.CRITICAL)

from deepsearcher.embedding import LiteLLMEmbedding


class TestLiteLLMEmbedding(unittest.TestCase):
    """Tests for the LiteLLMEmbedding class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_litellm = MagicMock()

        mock_data_item = {"embedding": [0.1] * 1536}
        self.mock_response = MagicMock()
        self.mock_response.data = [mock_data_item]
        self.mock_litellm.embedding.return_value = self.mock_response

        self.module_patcher = patch.dict("sys.modules", {"litellm": self.mock_litellm})
        self.module_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch.dict("os.environ", {}, clear=True):
            embedding = LiteLLMEmbedding()
            self.assertEqual(embedding.model, "text-embedding-ada-002")
            self.assertEqual(embedding.dim, 1536)
            self.assertIsNone(embedding.api_key)
            self.assertIsNone(embedding.api_base)

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {"LITELLM_API_KEY": "test-key", "LITELLM_API_BASE": "http://localhost:4000"},
        ):
            embedding = LiteLLMEmbedding()
            self.assertEqual(embedding.api_key, "test-key")
            self.assertEqual(embedding.api_base, "http://localhost:4000")

    def test_init_with_parameters(self):
        """Test initialization with parameters."""
        with patch.dict("os.environ", {}, clear=True):
            embedding = LiteLLMEmbedding(
                model="cohere/embed-english-v3.0",
                api_key="param-key",
                api_base="http://proxy:4000",
                dimension=1024,
            )
            self.assertEqual(embedding.model, "cohere/embed-english-v3.0")
            self.assertEqual(embedding.api_key, "param-key")
            self.assertEqual(embedding.api_base, "http://proxy:4000")
            self.assertEqual(embedding.dim, 1024)

    def test_embed_query(self):
        """Test embedding a single query."""
        with patch.dict("os.environ", {}, clear=True):
            embedding = LiteLLMEmbedding()

        result = embedding.embed_query("test query")

        self.mock_litellm.embedding.assert_called_once_with(
            model="text-embedding-ada-002",
            input=["test query"],
            drop_params=True,
        )
        self.assertEqual(result, [0.1] * 1536)

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        mock_data_items = [
            {"embedding": [0.1 * (i + 1)] * 1536} for i in range(3)
        ]
        self.mock_response.data = mock_data_items

        with patch.dict("os.environ", {}, clear=True):
            embedding = LiteLLMEmbedding()

        texts = ["text 1", "text 2", "text 3"]
        results = embedding.embed_documents(texts)

        self.mock_litellm.embedding.assert_called_once_with(
            model="text-embedding-ada-002",
            input=texts,
            drop_params=True,
        )
        self.assertEqual(len(results), 3)

    def test_embed_query_with_credentials(self):
        """Test embed_query passes api_key and api_base when set."""
        with patch.dict("os.environ", {}, clear=True):
            embedding = LiteLLMEmbedding(api_key="my-key", api_base="http://proxy:4000")

        embedding.embed_query("test")

        self.mock_litellm.embedding.assert_called_once_with(
            model="text-embedding-ada-002",
            input=["test"],
            drop_params=True,
            api_key="my-key",
            api_base="http://proxy:4000",
        )

    def test_embed_query_omits_credentials_when_not_set(self):
        """Test embed_query omits api_key and api_base when not set."""
        with patch.dict("os.environ", {}, clear=True):
            embedding = LiteLLMEmbedding()

        embedding.embed_query("test")

        call_kwargs = self.mock_litellm.embedding.call_args[1]
        self.assertNotIn("api_key", call_kwargs)
        self.assertNotIn("api_base", call_kwargs)

    def test_dimension_property(self):
        """Test the dimension property."""
        with patch.dict("os.environ", {}, clear=True):
            embedding = LiteLLMEmbedding()
            self.assertEqual(embedding.dimension, 1536)

            embedding = LiteLLMEmbedding(dimension=768)
            self.assertEqual(embedding.dimension, 768)


if __name__ == "__main__":
    unittest.main()
