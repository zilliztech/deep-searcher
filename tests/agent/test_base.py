import unittest

import numpy as np

from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM, ChatResponse
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult


class MockLLM(BaseLLM):
    """Mock LLM implementation for testing agents."""
    
    def __init__(self, predefined_responses=None):
        """
        Initialize the MockLLM.
        
        Args:
            predefined_responses: Dictionary mapping prompt substrings to responses
        """
        self.chat_called = False
        self.last_messages = None
        self.predefined_responses = predefined_responses or {}
    
    def chat(self, messages, **kwargs):
        """Mock implementation of chat that returns predefined responses or a default response."""
        self.chat_called = True
        self.last_messages = messages
        
        if self.predefined_responses:
            message_content = messages[0]["content"] if messages else ""
            for key, response in self.predefined_responses.items():
                if key in message_content:
                    return ChatResponse(content=response, total_tokens=10)
        
        return ChatResponse(content="This is a test answer", total_tokens=10)
    
    def literal_eval(self, text):
        """Mock implementation of literal_eval."""
        # Default implementation returns a list with test_collection
        # Override this in specific tests if needed
        if text.strip().startswith("[") and text.strip().endswith("]"):
            # Return the list as is if it's already in list format
            try:
                import ast
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                pass
                
        return ["test_collection"]


class MockEmbedding(BaseEmbedding):
    """Mock embedding model implementation for testing agents."""
    
    def __init__(self, dimension=8):
        """Initialize the MockEmbedding with a specific dimension."""
        self._dimension = dimension
    
    @property
    def dimension(self):
        """Return the dimension of the embedding model."""
        return self._dimension
    
    def embed_query(self, text):
        """Mock implementation that returns a random vector of the specified dimension."""
        return np.random.random(self._dimension).tolist()
    
    def embed_documents(self, documents):
        """Mock implementation that returns random vectors for each document."""
        return [np.random.random(self._dimension).tolist() for _ in documents]


class MockVectorDB(BaseVectorDB):
    """Mock vector database implementation for testing agents."""
    
    def __init__(self, collections=None):
        """
        Initialize the MockVectorDB.
        
        Args:
            collections: List of collection names to initialize with
        """
        self.default_collection = "test_collection"
        self.search_called = False
        self.insert_called = False
        self._collections = []
        
        if collections:
            for collection in collections:
                self._collections.append(
                    CollectionInfo(collection_name=collection, description=f"Test collection {collection}")
                )
        else:
            self._collections = [
                CollectionInfo(collection_name="test_collection", description="Test collection for testing")
            ]
    
    def search_data(self, collection, vector, top_k=10, **kwargs):
        """Mock implementation that returns test results."""
        self.search_called = True
        self.last_search_collection = collection
        self.last_search_vector = vector
        self.last_search_top_k = top_k
        self.last_search_kwargs = kwargs
        
        return [
            RetrievalResult(
                embedding=vector,
                text=f"Test result {i} for collection {collection}",
                reference=f"test_reference_{collection}_{i}",
                metadata={"a": i, "wider_text": f"Wider context for test result {i} in collection {collection}"}
            )
            for i in range(min(3, top_k))
        ]
    
    def insert_data(self, collection, chunks):
        """Mock implementation of insert_data."""
        self.insert_called = True
        self.last_insert_collection = collection
        self.last_insert_chunks = chunks
        return True
    
    def init_collection(self, dim, collection, **kwargs):
        """Mock implementation of init_collection."""
        return True
    
    def list_collections(self, dim=None):
        """Mock implementation that returns the list of collections."""
        return self._collections
    
    def clear_db(self, collection):
        """Mock implementation of clear_db."""
        return True


class BaseAgentTest(unittest.TestCase):
    """Base test class for agent tests with common setup."""
    
    def setUp(self):
        """Set up test fixtures for agent tests."""
        self.llm = MockLLM()
        self.embedding_model = MockEmbedding(dimension=8)
        self.vector_db = MockVectorDB()
