from unittest.mock import MagicMock

from deepsearcher.agent import NaiveRAG
from deepsearcher.vector_db.base import RetrievalResult
from tests.agent.test_base import BaseAgentTest


class TestNaiveRAG(BaseAgentTest):
    """Test class for NaiveRAG agent."""
    
    def setUp(self):
        """Set up test fixtures for NaiveRAG tests."""
        super().setUp()
        self.naive_rag = NaiveRAG(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            top_k=5,
            route_collection=True,
            text_window_splitter=True
        )
    
    def test_init(self):
        """Test the initialization of NaiveRAG."""
        self.assertEqual(self.naive_rag.llm, self.llm)
        self.assertEqual(self.naive_rag.embedding_model, self.embedding_model)
        self.assertEqual(self.naive_rag.vector_db, self.vector_db)
        self.assertEqual(self.naive_rag.top_k, 5)
        self.assertEqual(self.naive_rag.route_collection, True)
        self.assertEqual(self.naive_rag.text_window_splitter, True)
    
    def test_retrieve(self):
        """Test the retrieve method."""
        query = "Test query"
        
        # Mock the collection_router.invoke method
        self.naive_rag.collection_router.invoke = MagicMock(return_value=(["test_collection"], 5))
        
        results, tokens, metadata = self.naive_rag.retrieve(query)
        
        # Check if correct methods were called
        self.naive_rag.collection_router.invoke.assert_called_once()
        self.assertTrue(self.vector_db.search_called)
        
        # Check the results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)  # Should match our mock return of 3 results
        for result in results:
            self.assertIsInstance(result, RetrievalResult)
        
        # Check token count
        self.assertEqual(tokens, 5)  # From our mocked collection_router.invoke

    def test_retrieve_forwards_authorization_context(self):
        """Test retrieve passes caller context to routing and vector search."""
        query = "Test query"

        self.naive_rag.collection_router.invoke = MagicMock(return_value=(["test_collection"], 5))

        results, tokens, metadata = self.naive_rag.retrieve(
            query,
            authorized_collection_set=["test_collection"],
            tenant_id="tenant_a",
        )

        self.naive_rag.collection_router.invoke.assert_called_once_with(
            query=query,
            dim=self.embedding_model.dimension,
            authorized_collection_set=["test_collection"],
            tenant_id="tenant_a",
        )
        self.assertEqual(self.vector_db.last_search_kwargs["authorized_collection_set"], ["test_collection"])
        self.assertEqual(self.vector_db.last_search_kwargs["tenant_id"], "tenant_a")
        self.assertEqual(tokens, 5)
        self.assertEqual(len(results), 3)
    
    def test_retrieve_without_routing(self):
        """Test retrieve method with routing disabled."""
        self.naive_rag.route_collection = False
        query = "Test query without routing"
        
        results, tokens, metadata = self.naive_rag.retrieve(query)
        
        # Check that routing was not called
        self.assertTrue(self.vector_db.search_called)
        
        # Check the results
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, RetrievalResult)
        
        # Check token count
        self.assertEqual(tokens, 0)  # No tokens used for routing

    def test_retrieve_without_routing_filters_authorized_collections(self):
        """Test disabled routing still respects authorized collections."""
        self.vector_db._collections.append(
            self.vector_db._collections[0].__class__(
                collection_name="secret_collection",
                description="Secret collection",
            )
        )
        self.naive_rag.collection_router.all_collections = ["test_collection", "secret_collection"]
        self.naive_rag.route_collection = False
        query = "Test query without routing"

        results, tokens, metadata = self.naive_rag.retrieve(
            query,
            authorized_collection_set=["test_collection"],
        )

        self.assertEqual(self.vector_db.last_search_collection, "test_collection")
        self.assertEqual(self.vector_db.last_search_kwargs["authorized_collection_set"], ["test_collection"])
        self.assertEqual(tokens, 0)
        self.assertEqual(len(results), 3)
    
    def test_query(self):
        """Test the query method."""
        query = "Test query for full RAG"
        
        # Mock the retrieve method
        mock_results = [
            RetrievalResult(
                embedding=[0.1] * 8,
                text=f"Test result {i}",
                reference="test_reference",
                metadata={"a": i, "wider_text": f"Wider context for test result {i}"}
            )
            for i in range(3)
        ]
        self.naive_rag.retrieve = MagicMock(return_value=(mock_results, 5, {}))
        
        answer, retrieved_results, tokens = self.naive_rag.query(query)
        
        # Check if correct methods were called
        self.naive_rag.retrieve.assert_called_once_with(query)
        self.assertTrue(self.llm.chat_called)
        
        # Check the messages sent to LLM
        self.assertIn("content", self.llm.last_messages[0])
        self.assertIn(query, self.llm.last_messages[0]["content"])
        
        # Check the results
        self.assertEqual(answer, "This is a test answer")
        self.assertEqual(retrieved_results, mock_results)
        self.assertEqual(tokens, 15)  # 5 from retrieve + 10 from LLM
    
    def test_with_window_splitter_disabled(self):
        """Test with text window splitter disabled."""
        self.naive_rag.text_window_splitter = False
        query = "Test query with window splitter off"
        
        # Mock the retrieve method
        mock_results = [
            RetrievalResult(
                embedding=[0.1] * 8,
                text=f"Test result {i}",
                reference="test_reference",
                metadata={"a": i, "wider_text": f"Wider context for test result {i}"}
            )
            for i in range(3)
        ]
        self.naive_rag.retrieve = MagicMock(return_value=(mock_results, 5, {}))
        
        answer, retrieved_results, tokens = self.naive_rag.query(query)
        
        # Check that regular text is used instead of wider_text
        self.assertIn("Test result 0", self.llm.last_messages[0]["content"])
        self.assertNotIn("Wider context", self.llm.last_messages[0]["content"])


if __name__ == "__main__":
    import unittest
    unittest.main()
