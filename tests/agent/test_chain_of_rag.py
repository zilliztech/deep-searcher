from unittest.mock import MagicMock

from deepsearcher.agent import ChainOfRAG
from deepsearcher.llm.base import ChatResponse
from deepsearcher.vector_db.base import RetrievalResult
from tests.agent.test_base import BaseAgentTest


class TestChainOfRAG(BaseAgentTest):
    """Test class for ChainOfRAG agent."""
    
    def setUp(self):
        """Set up test fixtures for ChainOfRAG tests."""
        super().setUp()
        
        # Set up predefined responses for the LLM for exact prompt substrings
        self.llm.predefined_responses = {
            "previous queries and answers, generate a new simple follow-up question": "What is the significance of deep learning?",
            "Given the following documents, generate an appropriate answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            "given the following intermediate queries and answers, judge whether you have enough information": "Yes",
            "Given a list of agent indexes and corresponding descriptions": "1",
            "Given the following documents, select the ones that are support the Q-A pair": "[0, 1]",
            "Given the following intermediate queries and answers, generate a final answer": "Deep learning is an advanced subset of machine learning that uses neural networks with multiple layers."
        }
        
        self.chain_of_rag = ChainOfRAG(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            max_iter=3,
            early_stopping=True,
            route_collection=True,
            text_window_splitter=True
        )
    
    def test_init(self):
        """Test the initialization of ChainOfRAG."""
        self.assertEqual(self.chain_of_rag.llm, self.llm)
        self.assertEqual(self.chain_of_rag.embedding_model, self.embedding_model)
        self.assertEqual(self.chain_of_rag.vector_db, self.vector_db)
        self.assertEqual(self.chain_of_rag.max_iter, 3)
        self.assertEqual(self.chain_of_rag.early_stopping, True)
        self.assertEqual(self.chain_of_rag.route_collection, True)
        self.assertEqual(self.chain_of_rag.text_window_splitter, True)
    
    def test_reflect_get_subquery(self):
        """Test the _reflect_get_subquery method."""
        query = "What is deep learning?"
        intermediate_context = ["Previous query: What is AI?", "Previous answer: AI is artificial intelligence."]
        
        # Direct mock for this specific method
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content="What is the significance of deep learning?",
            total_tokens=10
        ))
        
        subquery, tokens = self.chain_of_rag._reflect_get_subquery(query, intermediate_context)
        
        self.assertEqual(subquery, "What is the significance of deep learning?")
        self.assertEqual(tokens, 10)
        self.assertTrue(self.llm.chat.called)
    
    def test_retrieve_and_answer(self):
        """Test the _retrieve_and_answer method."""
        query = "What is deep learning?"
        
        # Mock the collection_router.invoke method
        self.chain_of_rag.collection_router.invoke = MagicMock(return_value=(["test_collection"], 5))
        
        # Direct mock for this specific method
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            total_tokens=10
        ))
        
        answer, results, tokens = self.chain_of_rag._retrieve_and_answer(query)
        
        # Check if correct methods were called
        self.chain_of_rag.collection_router.invoke.assert_called_once()
        self.assertTrue(self.vector_db.search_called)
        
        # Check the results
        self.assertEqual(answer, "Deep learning is a subset of machine learning that uses neural networks with multiple layers.")
        self.assertEqual(tokens, 15)  # 5 from collection_router + 10 from LLM

    def test_retrieve_and_answer_forwards_authorization_context(self):
        """Test vector DB search receives caller authorization context."""
        query = "What is deep learning?"

        self.chain_of_rag.collection_router.invoke = MagicMock(return_value=(["test_collection"], 5))
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            total_tokens=10
        ))

        answer, results, tokens = self.chain_of_rag._retrieve_and_answer(
            query,
            authorized_collection_set=["test_collection"],
            tenant_id="tenant_a",
        )

        self.chain_of_rag.collection_router.invoke.assert_called_once_with(
            query=query,
            dim=self.embedding_model.dimension,
            authorized_collection_set=["test_collection"],
            tenant_id="tenant_a",
        )
        self.assertEqual(self.vector_db.last_search_kwargs["authorized_collection_set"], ["test_collection"])
        self.assertEqual(self.vector_db.last_search_kwargs["tenant_id"], "tenant_a")
        self.assertEqual(answer, "Deep learning is a subset of machine learning that uses neural networks with multiple layers.")
        self.assertEqual(len(results), 3)
        self.assertEqual(tokens, 15)
    
    def test_get_supported_docs(self):
        """Test the _get_supported_docs method."""
        results = [
            RetrievalResult(
                embedding=[0.1] * 8,
                text=f"Test result {i}",
                reference="test_reference",
                metadata={"a": i}
            )
            for i in range(3)
        ]
        
        query = "What is deep learning?"
        answer = "Deep learning is a subset of machine learning that uses neural networks with multiple layers."
        
        # Mock the literal_eval method to return indices as integers
        self.llm.literal_eval = MagicMock(return_value=[0, 1])
        
        supported_docs, tokens = self.chain_of_rag._get_supported_docs(results, query, answer)
        
        self.assertEqual(len(supported_docs), 2)  # Based on our mock response of [0, 1]
        self.assertEqual(tokens, 10)
    
    def test_check_has_enough_info(self):
        """Test the _check_has_enough_info method."""
        query = "What is deep learning?"
        intermediate_contexts = [
            "Intermediate query1: What is deep learning?",
            "Intermediate answer1: Deep learning is a subset of machine learning that uses neural networks with multiple layers."
        ]
        
        # Direct mock for this specific method
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content="Yes",
            total_tokens=10
        ))
        
        has_enough, tokens = self.chain_of_rag._check_has_enough_info(query, intermediate_contexts)
        
        self.assertTrue(has_enough)  # Based on our mock response of "Yes"
        self.assertEqual(tokens, 10)
    
    def test_retrieve(self):
        """Test the retrieve method."""
        query = "What is deep learning?"
        
        # Mock all the methods that retrieve calls
        self.chain_of_rag._reflect_get_subquery = MagicMock(return_value=("What is the significance of deep learning?", 5))
        self.chain_of_rag._retrieve_and_answer = MagicMock(
            return_value=("Deep learning is important in AI", [RetrievalResult(
                embedding=[0.1] * 8,
                text="Test result",
                reference="test_reference",
                metadata={"a": 1}
            )], 10)
        )
        self.chain_of_rag._get_supported_docs = MagicMock(return_value=([RetrievalResult(
            embedding=[0.1] * 8,
            text="Test result",
            reference="test_reference",
            metadata={"a": 1}
        )], 5))
        self.chain_of_rag._check_has_enough_info = MagicMock(return_value=(True, 5))
        
        results, tokens, metadata = self.chain_of_rag.retrieve(query)
        
        # Check if methods were called
        self.chain_of_rag._reflect_get_subquery.assert_called_once()
        self.chain_of_rag._retrieve_and_answer.assert_called_once()
        self.chain_of_rag._get_supported_docs.assert_called_once()
        
        # With early stopping, it should check if we have enough info
        self.chain_of_rag._check_has_enough_info.assert_called_once()
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(tokens, 25)  # 5 + 10 + 5 + 5
        self.assertIn("intermediate_context", metadata)
        
    def test_query(self):
        """Test the query method."""
        query = "What is deep learning?"
        
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
        
        self.chain_of_rag.retrieve = MagicMock(
            return_value=(retrieved_results, 20, {"intermediate_context": ["Some context"]})
        )
        
        # Direct mock for this specific method
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content="Deep learning is an advanced subset of machine learning that uses neural networks with multiple layers.",
            total_tokens=10
        ))
        
        answer, results, tokens = self.chain_of_rag.query(query)
        
        # Check if methods were called
        self.chain_of_rag.retrieve.assert_called_once_with(query)
        self.assertTrue(self.llm.chat.called)
        
        # Check results
        self.assertEqual(answer, "Deep learning is an advanced subset of machine learning that uses neural networks with multiple layers.")
        self.assertEqual(results, retrieved_results)
        self.assertEqual(tokens, 30)  # 20 from retrieve + 10 from LLM
    
    def test_format_retrieved_results(self):
        """Test the _format_retrieved_results method."""
        retrieved_results = [
            RetrievalResult(
                embedding=[0.1] * 8,
                text="Test result 1",
                reference="test_reference",
                metadata={"a": 1, "wider_text": "Wider context for test result 1"}
            ),
            RetrievalResult(
                embedding=[0.1] * 8,
                text="Test result 2",
                reference="test_reference",
                metadata={"a": 2, "wider_text": "Wider context for test result 2"}
            )
        ]
        
        # Test with text_window_splitter enabled
        self.chain_of_rag.text_window_splitter = True
        formatted = self.chain_of_rag._format_retrieved_results(retrieved_results)
        
        self.assertIn("Wider context for test result 1", formatted)
        self.assertIn("Wider context for test result 2", formatted)
        
        # Test with text_window_splitter disabled
        self.chain_of_rag.text_window_splitter = False
        formatted = self.chain_of_rag._format_retrieved_results(retrieved_results)
        
        self.assertIn("Test result 1", formatted)
        self.assertIn("Test result 2", formatted)
        self.assertNotIn("Wider context for test result 1", formatted)


if __name__ == "__main__":
    import unittest
    unittest.main()
