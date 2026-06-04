import logging
import os
import unittest
from unittest.mock import MagicMock, patch

logging.disable(logging.CRITICAL)

from deepsearcher.llm.base import ChatResponse
from deepsearcher.llm import LiteLLM


class TestLiteLLM(unittest.TestCase):
    """Tests for the LiteLLM LLM provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_litellm = MagicMock()

        self.mock_response = MagicMock()
        self.mock_choice = MagicMock()
        self.mock_message = MagicMock()
        self.mock_usage = MagicMock()

        self.mock_message.content = "Test response"
        self.mock_choice.message = self.mock_message
        self.mock_usage.total_tokens = 100

        self.mock_response.choices = [self.mock_choice]
        self.mock_response.usage = self.mock_usage
        self.mock_litellm.completion.return_value = self.mock_response

        self.module_patcher = patch.dict("sys.modules", {"litellm": self.mock_litellm})
        self.module_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.module_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM()
            self.assertEqual(llm.model, "gpt-4o-mini")
            self.assertIsNone(llm.api_key)
            self.assertIsNone(llm.api_base)

    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment variable."""
        with patch.dict(
            os.environ,
            {"LITELLM_API_KEY": "test-key", "LITELLM_API_BASE": "http://localhost:4000"},
        ):
            llm = LiteLLM()
            self.assertEqual(llm.api_key, "test-key")
            self.assertEqual(llm.api_base, "http://localhost:4000")

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key as parameter."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM(api_key="param-key", api_base="http://proxy:4000")
            self.assertEqual(llm.api_key, "param-key")
            self.assertEqual(llm.api_base, "http://proxy:4000")

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM(model="anthropic/claude-sonnet-4-20250514")
            self.assertEqual(llm.model, "anthropic/claude-sonnet-4-20250514")

    def test_chat_single_message(self):
        """Test chat with a single message."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM()

        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages)

        self.mock_litellm.completion.assert_called_once_with(
            model="gpt-4o-mini",
            messages=messages,
            drop_params=True,
        )

        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_multiple_messages(self):
        """Test chat with multiple messages."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM()

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = llm.chat(messages)

        self.mock_litellm.completion.assert_called_once_with(
            model="gpt-4o-mini",
            messages=messages,
            drop_params=True,
        )

        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.total_tokens, 100)

    def test_chat_with_api_key(self):
        """Test chat passes api_key and api_base when set."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM(api_key="my-key", api_base="http://proxy:4000")

        messages = [{"role": "user", "content": "Hello"}]
        llm.chat(messages)

        self.mock_litellm.completion.assert_called_once_with(
            model="gpt-4o-mini",
            messages=messages,
            drop_params=True,
            api_key="my-key",
            api_base="http://proxy:4000",
        )

    def test_chat_omits_credentials_when_not_set(self):
        """Test chat omits api_key and api_base when not set."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM()

        messages = [{"role": "user", "content": "Hello"}]
        llm.chat(messages)

        call_kwargs = self.mock_litellm.completion.call_args[1]
        self.assertNotIn("api_key", call_kwargs)
        self.assertNotIn("api_base", call_kwargs)

    def test_chat_with_error(self):
        """Test chat when an error occurs."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM()

        self.mock_litellm.completion.side_effect = Exception("LiteLLM API Error")

        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(Exception) as context:
            llm.chat(messages)

        self.assertEqual(str(context.exception), "LiteLLM API Error")

    def test_drop_params_always_true(self):
        """Test that drop_params=True is always passed."""
        with patch.dict("os.environ", {}, clear=True):
            llm = LiteLLM()

        llm.chat([{"role": "user", "content": "Hello"}])

        call_kwargs = self.mock_litellm.completion.call_args[1]
        self.assertTrue(call_kwargs["drop_params"])


if __name__ == "__main__":
    unittest.main()
