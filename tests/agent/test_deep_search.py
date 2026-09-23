import asyncio
from unittest.mock import MagicMock

from deepsearcher.agent import DeepSearch
from deepsearcher.vector_db.base import RetrievalResult
from tests.agent.test_base import BaseAgentTest


class TestDeepSearch(BaseAgentTest):
    """Test class for DeepSearch agent."""
    
    def setUp(self):
        """Set up test fixtures for DeepSearch tests."""
        super().setUp()
        
        # Set up predefined responses for the LLM for exact prompt substrings
        self.llm.predefined_responses = {
            "Original Question:": '["What is deep learning?", "How does deep learning work?", "What are applications of deep learning?"]',
            "Is the chunk helpful": "YES",
            "Respond exclusively in valid List": '["What are limitations of deep learning?"]',
            "You are a AI content analysis expert": "Deep learning is a subset of machine learning that uses neural networks with multiple layers."
        }
        
        self.deep_search = DeepSearch(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            max_iter=2,
            route_collection=True,
            text_window_splitter=True
        )
    
    def test_init(self):
        """Test the initialization of DeepSearch."""
        self.assertEqual(self.deep_search.llm, self.llm)
        self.assertEqual(self.deep_search.embedding_model, self.embedding_model)
        self.assertEqual(self.deep_search.vector_db, self.vector_db)
        self.assertEqual(self.deep_search.max_iter, 2)
        self.assertEqual(self.deep_search.route_collection, True)
        self.assertEqual(self.deep_search.text_window_splitter, True)
    
    def test_generate_sub_queries(self):
        """Test the _generate_sub_queries method."""
        query = "Tell me about deep learning"
        
        sub_queries, tokens = self.deep_search._generate_sub_queries(query)
        
        self.assertEqual(len(sub_queries), 3)
        self.assertEqual(sub_queries[0], "What is deep learning?")
        self.assertEqual(sub_queries[1], "How does deep learning work?")
        self.assertEqual(sub_queries[2], "What are applications of deep learning?")
        self.assertEqual(tokens, 10)
        self.assertTrue(self.llm.chat_called)
    
    def test_search_chunks_from_vectordb(self):
        """Test the _search_chunks_from_vectordb method."""
        query = "What is deep learning?"
        sub_queries = ["What is deep learning?", "How does deep learning work?"]
        
        # Mock the collection_router.invoke method
        self.deep_search.collection_router.invoke = MagicMock(return_value=(["test_collection"], 5))
        
        # Run the async method using asyncio.run
        results, tokens = asyncio.run(
            self.deep_search._search_chunks_from_vectordb(query, sub_queries)
        )
        
        # Check if correct methods were called
        self.deep_search.collection_router.invoke.assert_called_once()
        self.assertTrue(self.vector_db.search_called)
        self.assertTrue(self.llm.chat_called)
        
        # With our mock returning "YES" for RERANK_PROMPT, all chunks should be accepted
        self.assertEqual(len(results), 3)  # 3 mock results from MockVectorDB
        self.assertEqual(tokens, 35)  # 5 from collection_router + 10*3 from LLM calls for reranking

    def test_search_chunks_from_vectordb_forwards_authorization_context(self):
        """Test vector DB search receives caller authorization context."""
        query = "What is deep learning?"
        sub_queries = ["What is deep learning?"]

        self.deep_search.collection_router.invoke = MagicMock(return_value=(["test_collection"], 5))

        results, tokens = asyncio.run(
            self.deep_search._search_chunks_from_vectordb(
                query,
                sub_queries,
                authorized_collection_set=["test_collection"],
                tenant_id="tenant_a",
            )
        )

        self.deep_search.collection_router.invoke.assert_called_once_with(
            query=query,
            dim=self.embedding_model.dimension,
            authorized_collection_set=["test_collection"],
            tenant_id="tenant_a",
        )
        self.assertEqual(self.vector_db.last_search_kwargs["authorized_collection_set"], ["test_collection"])
        self.assertEqual(self.vector_db.last_search_kwargs["tenant_id"], "tenant_a")
        self.assertEqual(len(results), 3)
        self.assertEqual(tokens, 35)
    
    def test_generate_gap_queries(self):
        """Test the _generate_gap_queries method."""
        query = "Tell me about deep learning"
        all_sub_queries = ["What is deep learning?", "How does deep learning work?"]
        all_chunks = [
            RetrievalResult(
                embedding=[0.1] * 8,
                text="Deep learning is a subset of machine learning",
                reference="test_reference",
                metadata={"a": 1}
            ),
            RetrievalResult(
                embedding=[0.1] * 8,
                text="Deep learning uses neural networks",
                reference="test_reference",
                metadata={"a": 2}
            )
        ]
        
        gap_queries, tokens = self.deep_search._generate_gap_queries(query, all_sub_queries, all_chunks)
        
        self.assertEqual(len(gap_queries), 1)
        self.assertEqual(gap_queries[0], "What are limitations of deep learning?")
        self.assertEqual(tokens, 10)
    
    def test_retrieve(self):
        """Test the retrieve method."""
        query = "Tell me about deep learning"
        
        # Mock async method to run synchronously
        async def mock_async_retrieve(*args, **kwargs):
            # Create some test results
            results = [
                RetrievalResult(
                    embedding=[0.1] * 8,
                    text="Deep learning is a subset of machine learning",
                    reference="test_reference",
                    metadata={"a": 1}
                ),
                RetrievalResult(
                    embedding=[0.1] * 8,
                    text="Deep learning uses neural networks",
                    reference="test_reference",
                    metadata={"a": 2}
                )
            ]
            # Return the results, token count, and additional info
            return results, 30, {"all_sub_queries": ["What is deep learning?", "How does deep learning work?"]}
        
        # Replace the async method with our mock
        self.deep_search.async_retrieve = mock_async_retrieve
        
        results, tokens, metadata = self.deep_search.retrieve(query)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(tokens, 30)
        self.assertIn("all_sub_queries", metadata)
        self.assertEqual(len(metadata["all_sub_queries"]), 2)
    
    def test_async_retrieve(self):
        """Test the async_retrieve method."""
        query = "Tell me about deep learning"
        
        # Create mock results
        mock_results = [
            RetrievalResult(
                embedding=[0.1] * 8,
                text="Deep learning is a subset of machine learning",
                reference="test_reference",
                metadata={"a": 1}
            )
        ]
        
        # Create a mock async_retrieve result
        mock_retrieve_result = (
            mock_results, 
            20, 
            {"all_sub_queries": ["What is deep learning?", "How does deep learning work?"]}
        )
        
        # Mock the async_retrieve method
        async def mock_async_retrieve(*args, **kwargs):
            return mock_retrieve_result
            
        self.deep_search.async_retrieve = mock_async_retrieve
        
        # Run the async method using asyncio.run
        results, tokens, metadata = asyncio.run(self.deep_search.async_retrieve(query))
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(tokens, 20)
        self.assertIn("all_sub_queries", metadata)
    
    def test_query(self):
        """Test the query method."""
        query = "Tell me about deep learning"
        
        # Mock the retrieve method
        retrieved_results = [
            RetrievalResult(
                embedding=[0.1] * 8,
                text=f"Test result {i}",
                reference="test_reference",
                metadata={"a": i, "wider_text": f"Wider context for test result {i}"}
            )
            for i in range(3)
        ]
        
        self.deep_search.retrieve = MagicMock(
            return_value=(retrieved_results, 20, {"all_sub_queries": ["What is deep learning?"]})
        )
        
        answer, results, tokens = self.deep_search.query(query)
        
        # Check if methods were called
        self.deep_search.retrieve.assert_called_once_with(query)
        self.assertTrue(self.llm.chat_called)
        
        # Check results
        self.assertEqual(answer, "Deep learning is a subset of machine learning that uses neural networks with multiple layers.")
        self.assertEqual(results, retrieved_results)
        self.assertEqual(tokens, 30)  # 20 from retrieve + 10 from LLM
    
    def test_query_no_results(self):
        """Test the query method when no results are found."""
        query = "Tell me about deep learning"
        
        # Mock the retrieve method to return no results
        self.deep_search.retrieve = MagicMock(
            return_value=([], 10, {"all_sub_queries": ["What is deep learning?"]})
        )
        
        answer, results, tokens = self.deep_search.query(query)
        
        # Should return a message saying no results found
        self.assertIn("No relevant information found", answer)
        self.assertEqual(results, [])
        self.assertEqual(tokens, 10)  # Only tokens from retrieve
    
    def test_format_chunk_texts(self):
        """Test the _format_chunk_texts method."""
        chunk_texts = ["Text 1", "Text 2", "Text 3"]
        
        formatted = self.deep_search._format_chunk_texts(chunk_texts)
        
        self.assertIn("<chunk_0>", formatted)
        self.assertIn("Text 1", formatted)
        self.assertIn("<chunk_1>", formatted)
        self.assertIn("Text 2", formatted)
        self.assertIn("<chunk_2>", formatted)
        self.assertIn("Text 3", formatted)


if __name__ == "__main__":
    import unittest
    unittest.main()
