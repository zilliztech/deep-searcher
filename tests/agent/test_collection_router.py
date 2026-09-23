from unittest.mock import MagicMock, patch

from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.llm.base import ChatResponse
from deepsearcher.vector_db.base import CollectionInfo
from tests.agent.test_base import BaseAgentTest


class TestCollectionRouter(BaseAgentTest):
    """Test class for CollectionRouter."""
    
    def setUp(self):
        """Set up test fixtures for CollectionRouter tests."""
        super().setUp()
        
        # Create mock collections
        self.collection_infos = [
            CollectionInfo(collection_name="books", description="Collection of book summaries"),
            CollectionInfo(collection_name="science", description="Scientific articles and papers"),
            CollectionInfo(collection_name="news", description="Recent news articles")
        ]
        
        # Configure vector_db mock
        self.vector_db.list_collections = MagicMock(return_value=self.collection_infos)
        self.vector_db.default_collection = "books"
        
        # Create the CollectionRouter
        self.collection_router = CollectionRouter(
            llm=self.llm,
            vector_db=self.vector_db,
            dim=8
        )
    
    def test_init(self):
        """Test the initialization of CollectionRouter."""
        self.assertEqual(self.collection_router.llm, self.llm)
        self.assertEqual(self.collection_router.vector_db, self.vector_db)
        self.assertEqual(
            self.collection_router.all_collections, 
            ["books", "science", "news"]
        )
    
    def test_invoke_with_multiple_collections(self):
        """Test the invoke method with multiple collections."""
        query = "What are the latest scientific breakthroughs?"
        
        # Mock LLM to return specific collections based on query
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content='["science", "news"]',
            total_tokens=10
        ))
        
        # Disable log output for testing
        with patch('deepsearcher.utils.log.color_print'):
            selected_collections, tokens = self.collection_router.invoke(query, dim=8)
        
        # Check results
        self.assertTrue("science" in selected_collections)
        self.assertTrue("news" in selected_collections)
        self.assertTrue("books" in selected_collections)  # Default collection is always included
        self.assertEqual(tokens, 10)
        
        # Verify that the LLM was called with the right prompt
        self.llm.chat.assert_called_once()
        self.assertIn(query, self.llm.chat.call_args[1]["messages"][0]["content"])
        self.assertIn("collection_name", self.llm.chat.call_args[1]["messages"][0]["content"])
    
    def test_invoke_with_empty_response(self):
        """Test the invoke method when LLM returns an empty list."""
        query = "Something completely unrelated"
        
        # Mock LLM to return empty list
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content='[]',
            total_tokens=5
        ))
        
        # Disable log output for testing
        with patch('deepsearcher.utils.log.color_print'):
            selected_collections, tokens = self.collection_router.invoke(query, dim=8)
        
        # Only default collection should be included
        self.assertEqual(len(selected_collections), 1)
        self.assertEqual(selected_collections[0], "books")
        self.assertEqual(tokens, 5)
    
    def test_invoke_with_no_collections(self):
        """Test the invoke method when no collections are available."""
        query = "Test query"
        
        # Mock vector_db to return empty list
        self.vector_db.list_collections = MagicMock(return_value=[])
        
        # Disable log warnings for testing
        with patch('deepsearcher.utils.log.warning'):
            with patch('deepsearcher.utils.log.color_print'):
                selected_collections, tokens = self.collection_router.invoke(query, dim=8)
        
        # Should return empty list and zero tokens
        self.assertEqual(selected_collections, [])
        self.assertEqual(tokens, 0)
    
    def test_invoke_with_single_collection(self):
        """Test the invoke method when only one collection is available."""
        query = "Test query"
        
        # Create a fresh mock for llm.chat to verify it's not called
        mock_chat = MagicMock(return_value=ChatResponse(content='[]', total_tokens=0))
        self.llm.chat = mock_chat
        
        # Mock vector_db to return single collection
        single_collection = [CollectionInfo(collection_name="single", description="The only collection")]
        self.vector_db.list_collections = MagicMock(return_value=single_collection)
        
        # Disable log output for testing
        with patch('deepsearcher.utils.log.color_print'):
            selected_collections, tokens = self.collection_router.invoke(query, dim=8)
        
        # Should return the only collection without calling LLM
        self.assertEqual(selected_collections, ["single"])
        self.assertEqual(tokens, 0)
        mock_chat.assert_not_called()
    
    def test_invoke_with_no_description(self):
        """Test the invoke method when a collection has no description."""
        query = "Test query"
        
        # Create collections with one having no description
        collections_with_no_desc = [
            CollectionInfo(collection_name="with_desc", description="Has description"),
            CollectionInfo(collection_name="no_desc", description="")
        ]
        self.vector_db.list_collections = MagicMock(return_value=collections_with_no_desc)
        self.vector_db.default_collection = "with_desc"
        
        # Mock LLM to return only the first collection
        self.llm.chat = MagicMock(return_value=ChatResponse(
            content='["with_desc"]',
            total_tokens=5
        ))
        
        # Disable log output for testing
        with patch('deepsearcher.utils.log.color_print'):
            selected_collections, tokens = self.collection_router.invoke(query, dim=8)
        
        # Both collections should be included (one from LLM, one with no description)
        self.assertEqual(set(selected_collections), {"with_desc", "no_desc"})
        self.assertEqual(tokens, 5)

    def test_invoke_filters_authorized_collections(self):
        """Test that routing only considers collections authorized for the caller."""
        query = "quarterly revenue"

        self.llm.chat = MagicMock(return_value=ChatResponse(
            content='["science", "news"]',
            total_tokens=10
        ))

        with patch('deepsearcher.utils.log.color_print'):
            selected_collections, tokens = self.collection_router.invoke(
                query,
                dim=8,
                authorized_collection_set=["news"],
            )

        self.assertEqual(selected_collections, ["news"])
        self.assertEqual(tokens, 0)
        self.llm.chat.assert_not_called()

    def test_invoke_filters_llm_selected_unauthorized_collections(self):
        """Test that LLM-selected collections are constrained by authorization."""
        query = "quarterly revenue"

        self.llm.chat = MagicMock(return_value=ChatResponse(
            content='["science", "news"]',
            total_tokens=10
        ))

        with patch('deepsearcher.utils.log.color_print'):
            selected_collections, tokens = self.collection_router.invoke(
                query,
                dim=8,
                authorized_collection_set=["books", "news"],
            )

        self.assertEqual(set(selected_collections), {"books", "news"})
        self.assertNotIn("science", selected_collections)
        self.assertEqual(tokens, 10)


if __name__ == "__main__":
    import unittest
    unittest.main()
