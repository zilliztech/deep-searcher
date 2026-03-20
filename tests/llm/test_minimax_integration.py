"""
Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable.
Run with: MINIMAX_API_KEY=your_key python -m pytest tests/llm/test_minimax_integration.py -v
"""

import os
import unittest

import pytest


@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY environment variable not set",
)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests for the MiniMax LLM provider."""

    def test_chat_basic(self):
        """Test basic chat completion with MiniMax API."""
        from deepsearcher.llm import MiniMax

        llm = MiniMax(model="MiniMax-M2.7")
        messages = [{"role": "user", "content": "Say hello in one word."}]
        response = llm.chat(messages)

        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 0)
        self.assertGreater(response.total_tokens, 0)

    def test_chat_with_system_message(self):
        """Test chat with system message."""
        from deepsearcher.llm import MiniMax

        llm = MiniMax(model="MiniMax-M2.7")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        response = llm.chat(messages)

        self.assertIsNotNone(response.content)
        self.assertIn("4", response.content)

    def test_chat_m25_highspeed(self):
        """Test chat with MiniMax-M2.5-highspeed model."""
        from deepsearcher.llm import MiniMax

        llm = MiniMax(model="MiniMax-M2.5-highspeed")
        messages = [{"role": "user", "content": "Say 'test' and nothing else."}]
        response = llm.chat(messages)

        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 0)


if __name__ == "__main__":
    unittest.main()
