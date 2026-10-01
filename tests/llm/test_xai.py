import unittest
from unittest.mock import patch, MagicMock
import os
import logging

# Disable logging for tests
logging.disable(logging.CRITICAL)

from deepsearcher.llm import XAI
from deepsearcher.llm.base import ChatResponse


class TestXAI(unittest.TestCase):
    """Tests for the X.AI LLM provider."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock module and components
        self.mock_openai = MagicMock()
        self.mock_client = MagicMock()
        self.mock_chat = MagicMock()
        self.mock_completions = MagicMock()
        
        # Set up the mock module structure
        self.mock_openai.OpenAI = MagicMock(return_value=self.mock_client)
        self.mock_client.chat = self.mock_chat
        self.mock_chat.completions = self.mock_completions
        
        # Set up mock response
        self.mock_response = MagicMock()
        self.mock_choice = MagicMock()
        self.mock_message = MagicMock()
        self.mock_usage = MagicMock()
        
        self.mock_message.content = "Test response"
        self.mock_choice.message = self.mock_message
        self.mock_usage.total_tokens = 100
        
        self.mock_response.choices = [self.mock_choice]
        self.mock_response.usage = self.mock_usage
        self.mock_completions.create.return_value = self.mock_response

        # Create the module patcher
        self.module_patcher = patch.dict('sys.modules', {'openai': self.mock_openai})
        self.module_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            llm = XAI()
            # Check that OpenAI client was initialized correctly
            self.mock_openai.OpenAI.assert_called_once_with(
                api_key=None,
                base_url="https://api.x.ai/v1"
            )
            
            # Check default model
            self.assertEqual(llm.model, "grok-4")

    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment variable."""
        api_key = "test_api_key_from_env"
        with patch.dict(os.environ, {"XAI_API_KEY": api_key}):
            llm = XAI()
            self.mock_openai.OpenAI.assert_called_with(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key as parameter."""
        with patch.dict('os.environ', {}, clear=True):
            api_key = "test_api_key_param"
            llm = XAI(api_key=api_key)
            self.mock_openai.OpenAI.assert_called_with(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict('os.environ', {}, clear=True):
            model = "grok-1"
            llm = XAI(model=model)
            self.assertEqual(llm.model, model)

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        # Clear environment variables temporarily
        with patch.dict('os.environ', {}, clear=True):
            base_url = "https://custom.x.ai"
            llm = XAI(base_url=base_url)
            self.mock_openai.OpenAI.assert_called_with(
                api_key=None,
                base_url=base_url
            )

    def test_chat_single_message(self):
        """Test chat with a single message."""
        # Create XAI instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = XAI()
            
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages)

        # Check that completions.create was called correctly
        self.mock_completions.create.assert_called_once()
        call_args = self.mock_completions.create.call_args
        self.assertEqual(call_args[1]["model"], "grok-4")
        self.assertEqual(call_args[1]["messages"], messages)

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_multiple_messages(self):
        """Test chat with multiple messages."""
        # Create XAI instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = XAI()
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        response = llm.chat(messages)

        # Check that completions.create was called correctly
        self.mock_completions.create.assert_called_once()
        call_args = self.mock_completions.create.call_args
        self.assertEqual(call_args[1]["model"], "grok-4")
        self.assertEqual(call_args[1]["messages"], messages)

        # Check response
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_with_error(self):
        """Test chat when an error occurs."""
        # Create XAI instance with mocked environment
        with patch.dict('os.environ', {}, clear=True):
            llm = XAI()
            
        # Mock an error response
        self.mock_completions.create.side_effect = Exception("XAI API Error")

        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(Exception) as context:
            llm.chat(messages)

        self.assertEqual(str(context.exception), "XAI API Error")


if __name__ == "__main__":
    unittest.main() 
