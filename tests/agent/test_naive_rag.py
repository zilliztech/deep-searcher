from unittest.mock import MagicMock

from deepsearcher.agent import NaiveRAG
from deepsearcher.vector_db.base import RetrievalResult

from tests.agent.test_base import BaseAgentTest, MockVectorDB


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
    
    def test_retrieve_without_routing_at_construction(self):
        """Disabling routing at construction still retrieves existing documents."""
        agent = NaiveRAG(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            route_collection=False,
        )

        results, tokens, metadata = agent.retrieve("Test query without routing")

        self.assertEqual(
            [result.reference for result in results],
            [f"test_reference_test_collection_{i}" for i in range(3)],
        )
        self.assertEqual(tokens, 0)
        self.assertEqual(metadata, {})
        self.assertFalse(self.llm.chat_called)

    def test_retrieve_multiple_collections_without_routing(self):
        """All collections are searched without an LLM selection step."""
        vector_db = MockVectorDB(collections=["first", "second"])
        agent = NaiveRAG(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vector_db=vector_db,
            top_k=6,
            route_collection=False,
        )

        results, tokens, metadata = agent.retrieve("Test query without routing")

        self.assertEqual(
            [result.reference for result in results],
            [
                f"test_reference_{collection}_{i}"
                for collection in ["first", "second"]
                for i in range(3)
            ],
        )
        self.assertEqual(tokens, 0)
        self.assertEqual(metadata, {})
        self.assertFalse(self.llm.chat_called)

    def test_retrieve_empty_database_without_routing(self):
        """An empty database returns an empty result without routing or search."""
        self.vector_db._collections = []
        agent = NaiveRAG(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            route_collection=False,
        )

        self.assertEqual(agent.retrieve("Test query without routing"), ([], 0, {}))
        self.assertFalse(self.vector_db.search_called)
        self.assertFalse(self.llm.chat_called)

    def test_query_without_routing_at_construction(self):
        """The query API retrieves evidence and only calls the LLM for the answer."""
        self.llm.chat = MagicMock(wraps=self.llm.chat)
        agent = NaiveRAG(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            route_collection=False,
        )

        answer, results, tokens = agent.query("Test query without routing")

        self.assertEqual(answer, "This is a test answer")
        self.assertEqual(
            [result.reference for result in results],
            [f"test_reference_test_collection_{i}" for i in range(3)],
        )
        self.assertEqual(tokens, 10)
        self.llm.chat.assert_called_once()
        prompt = self.llm.chat.call_args.args[0][0]["content"]
        for i in range(3):
            self.assertIn(f"Wider context for test result {i} in collection test_collection", prompt)

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