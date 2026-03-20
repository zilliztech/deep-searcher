"""
Integration tests for MiniMax Embedding provider.

These tests require a valid MINIMAX_API_KEY environment variable.
Run with: MINIMAX_API_KEY=your_key python -m pytest tests/embedding/test_minimax_embedding_integration.py -v
"""

import os
import unittest

import pytest


@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY environment variable not set",
)
class TestMiniMaxEmbeddingIntegration(unittest.TestCase):
    """Integration tests for the MiniMax embedding provider."""

    def test_embed_query(self):
        """Test embedding a single query."""
        from deepsearcher.embedding import MiniMaxEmbedding

        embedding = MiniMaxEmbedding()
        result = embedding.embed_query("Hello world")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1536)
        self.assertIsInstance(result[0], float)

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        from deepsearcher.embedding import MiniMaxEmbedding

        embedding = MiniMaxEmbedding()
        texts = ["First document", "Second document"]
        results = embedding.embed_documents(texts)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(len(result), 1536)

    def test_dimension_property(self):
        """Test the dimension property matches actual embeddings."""
        from deepsearcher.embedding import MiniMaxEmbedding

        embedding = MiniMaxEmbedding()
        result = embedding.embed_query("Test")

        self.assertEqual(len(result), embedding.dimension)


if __name__ == "__main__":
    unittest.main()
