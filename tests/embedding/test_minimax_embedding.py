import unittest
import os
from unittest.mock import patch, MagicMock

import requests
from deepsearcher.embedding import MiniMaxEmbedding


class TestMiniMaxEmbedding(unittest.TestCase):
    """Tests for the MiniMaxEmbedding class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create patches for requests
        self.requests_patcher = patch('requests.request')
        self.mock_request = self.requests_patcher.start()

        # Set up mock response
        self.mock_response = MagicMock()
        self.mock_response.json.return_value = {
            'vectors': [
                [0.1] * 1536  # embo-01 has 1536 dimensions
            ]
        }
        self.mock_response.raise_for_status = MagicMock()
        self.mock_request.return_value = self.mock_response

    def tearDown(self):
        """Clean up test fixtures."""
        self.requests_patcher.stop()

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        embedding = MiniMaxEmbedding()

        self.assertEqual(embedding.model, 'embo-01')
        self.assertEqual(embedding.api_key, 'fake-api-key')
        self.assertEqual(embedding.batch_size, 32)

    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        embedding = MiniMaxEmbedding(api_key='test-api-key')

        self.assertEqual(embedding.api_key, 'test-api-key')

    @patch.dict('os.environ', {}, clear=True)
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with self.assertRaises(RuntimeError):
            MiniMaxEmbedding()

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        embedding = MiniMaxEmbedding()

        query = "This is a test query"
        result = embedding.embed_query(query)

        # Verify that request was called with type="query"
        self.mock_request.assert_called_once_with(
            'POST',
            'https://api.minimax.io/v1/embeddings',
            json={
                'model': 'embo-01',
                'texts': [query],
                'type': 'query'
            },
            headers={
                'Authorization': 'Bearer fake-api-key',
                'Content-Type': 'application/json'
            }
        )

        self.assertEqual(result, [0.1] * 1536)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        embedding = MiniMaxEmbedding()

        texts = ["text 1", "text 2", "text 3"]

        # Set up mock response for multiple documents
        self.mock_response.json.return_value = {
            'vectors': [
                [0.1 * (i + 1)] * 1536
                for i in range(3)
            ]
        }

        results = embedding.embed_documents(texts)

        # Verify that request was called with type="db"
        self.mock_request.assert_called_once_with(
            'POST',
            'https://api.minimax.io/v1/embeddings',
            json={
                'model': 'embo-01',
                'texts': texts,
                'type': 'db'
            },
            headers={
                'Authorization': 'Bearer fake-api-key',
                'Content-Type': 'application/json'
            }
        )

        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result, [0.1 * (i + 1)] * 1536)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents_with_batching(self):
        """Test embedding documents with batching."""
        embedding = MiniMaxEmbedding()

        texts = ["text " + str(i) for i in range(50)]  # More than batch_size

        def mock_batch_response(*args, **kwargs):
            batch_input = kwargs['json']['texts']
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                'vectors': [
                    [0.1] * 1536
                    for _ in range(len(batch_input))
                ]
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        self.mock_request.side_effect = mock_batch_response

        results = embedding.embed_documents(texts)

        # Check that request was called multiple times
        self.assertTrue(self.mock_request.call_count > 1)

        # Check the results
        self.assertEqual(len(results), 50)
        for result in results:
            self.assertEqual(result, [0.1] * 1536)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'fake-api-key'}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        embedding = MiniMaxEmbedding()

        self.assertEqual(embedding.dimension, 1536)

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_query_uses_query_type(self):
        """Test that embed_query uses type='query' for retrieval."""
        embedding = MiniMaxEmbedding()
        embedding.embed_query("search query")

        call_args = self.mock_request.call_args
        self.assertEqual(call_args[1]['json']['type'], 'query')

    @patch.dict('os.environ', {'MINIMAX_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents_uses_db_type(self):
        """Test that embed_documents uses type='db' for storage."""
        embedding = MiniMaxEmbedding()
        embedding.embed_documents(["doc text"])

        call_args = self.mock_request.call_args
        self.assertEqual(call_args[1]['json']['type'], 'db')


if __name__ == "__main__":
    unittest.main()
