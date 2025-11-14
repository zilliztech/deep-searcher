import unittest
import os
from unittest.mock import patch, MagicMock

import requests
from deepsearcher.embedding import JiekouAIEmbedding


class TestJiekouAIEmbedding(unittest.TestCase):
    """Tests for the JiekouAIEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patches for requests
        self.requests_patcher = patch('requests.request')
        self.mock_request = self.requests_patcher.start()
        
        # Set up mock response
        self.mock_response = MagicMock()
        self.mock_response.json.return_value = {
            'data': [
                {'index': 0, 'embedding': [0.1] * 1024}  # baai/bge-m3 has 1024 dimensions
            ]
        }
        self.mock_response.raise_for_status = MagicMock()
        self.mock_request.return_value = self.mock_response
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.requests_patcher.stop()
    
    @patch.dict('os.environ', {'JIEKOU_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Create the embedder
        embedding = JiekouAIEmbedding()
        
        # Check attributes
        self.assertEqual(embedding.model, 'baai/bge-m3')
        self.assertEqual(embedding.api_key, 'fake-api-key')
        self.assertEqual(embedding.batch_size, 32)
    
    @patch.dict('os.environ', {'JIEKOU_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_model(self):
        """Test initialization with specified model."""
        # Initialize with a different model
        embedding = JiekouAIEmbedding(model='baai/bge-m3')
        
        # Check attributes
        self.assertEqual(embedding.model, 'baai/bge-m3')
        self.assertEqual(embedding.dimension, 1024)
    
    @patch.dict('os.environ', {'JIEKOU_API_KEY': 'fake-api-key'}, clear=True)
    def test_init_with_model_name(self):
        """Test initialization with model_name parameter."""
        # Initialize with model_name
        embedding = JiekouAIEmbedding(model_name='baai/bge-m3')
        
        # Check attributes
        self.assertEqual(embedding.model, 'baai/bge-m3')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        # Initialize with API key
        embedding = JiekouAIEmbedding(api_key='test-api-key')
        
        # Check that the API key was set correctly
        self.assertEqual(embedding.api_key, 'test-api-key')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with self.assertRaises(RuntimeError):
            JiekouAIEmbedding()
    
    @patch.dict('os.environ', {'JIEKOU_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create the embedder
        embedding = JiekouAIEmbedding()
        
        # Create a test query
        query = "This is a test query"
        
        # Call the method
        result = embedding.embed_query(query)
        
        # Verify that request was called correctly
        self.mock_request.assert_called_once_with(
            'POST',
            'https://api.jiekou.ai/openai/v1/embeddings',
            json={
                'model': 'baai/bge-m3',
                'input': [query]
            },
            headers={
                'Authorization': 'Bearer fake-api-key',
                'Content-Type': 'application/json'
            }
        )
        
        # Check the result
        self.assertEqual(result, [0.1] * 1024)
    
    @patch.dict('os.environ', {'JIEKOU_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create the embedder
        embedding = JiekouAIEmbedding()
        
        # Create test documents
        texts = ["text 1", "text 2", "text 3"]
        
        # Set up mock response for multiple documents
        self.mock_response.json.return_value = {
            'data': [
                {'index': i, 'embedding': [0.1 * (i + 1)] * 1024}
                for i in range(3)
            ]
        }
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Verify that request was called correctly
        self.mock_request.assert_called_once_with(
            'POST',
            'https://api.jiekou.ai/openai/v1/embeddings',
            json={
                'model': 'baai/bge-m3',
                'input': texts
            },
            headers={
                'Authorization': 'Bearer fake-api-key',
                'Content-Type': 'application/json'
            }
        )
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result, [0.1 * (i + 1)] * 1024)
    
    @patch.dict('os.environ', {'JIEKOU_API_KEY': 'fake-api-key'}, clear=True)
    def test_embed_documents_with_batching(self):
        """Test embedding documents with batching."""
        # Create the embedder
        embedding = JiekouAIEmbedding()
        
        # Create test documents
        texts = ["text " + str(i) for i in range(50)]  # More than batch_size
        
        # Set up mock response for batched documents
        def mock_batch_response(*args, **kwargs):
            batch_input = kwargs['json']['input']
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                'data': [
                    {'index': i, 'embedding': [0.1] * 1024}
                    for i in range(len(batch_input))
                ]
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp
        
        self.mock_request.side_effect = mock_batch_response
        
        # Call the method
        results = embedding.embed_documents(texts)
        
        # Check that request was called multiple times
        self.assertTrue(self.mock_request.call_count > 1)
        
        # Check the results
        self.assertEqual(len(results), 50)
        for result in results:
            self.assertEqual(result, [0.1] * 1024)
    
    @patch.dict('os.environ', {'JIEKOU_API_KEY': 'fake-api-key'}, clear=True)
    def test_dimension_property(self):
        """Test the dimension property."""
        # Create the embedder
        embedding = JiekouAIEmbedding()
        
        # For baai/bge-m3
        self.assertEqual(embedding.dimension, 1024)


if __name__ == "__main__":
    unittest.main() 