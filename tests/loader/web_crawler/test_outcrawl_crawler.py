import os
import unittest
from unittest.mock import MagicMock, patch

import requests
from langchain_core.documents import Document

from deepsearcher.loader.web_crawler import OutcrawlCrawler


class TestOutcrawlCrawler(unittest.TestCase):
    """Tests for the OutcrawlCrawler class."""

    @patch.dict(os.environ, {"OUTCRAWL_API_KEY": "fake-key"})
    def test_init_with_key(self):
        """Test initialization with API key in environment."""
        crawler = OutcrawlCrawler()
        self.assertEqual(crawler.outcrawl_api_key, "fake-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_key(self):
        """Test initialization without API key raises ValueError."""
        with self.assertRaises(ValueError):
            OutcrawlCrawler()

    @patch.dict(os.environ, {"OUTCRAWL_API_KEY": "fake-key"})
    @patch("requests.post")
    def test_crawl_url(self, mock_post):
        """Test crawling a URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "url": "https://example.com/",
            "statusCode": 200,
            "markdown": "# Markdown Content\nThis is a test.",
            "metadata": {"title": "Example Domain", "language": "en"},
        }
        mock_post.return_value = mock_response

        crawler = OutcrawlCrawler()
        url = "https://example.com"
        documents = crawler.crawl_url(url)

        mock_post.assert_called_once_with(
            "https://api.outcrawl.ai/v1/scrape",
            headers={
                "Authorization": "Bearer fake-key",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown"]},
            timeout=120,
        )

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertIsInstance(document, Document)
        self.assertEqual(document.page_content, "# Markdown Content\nThis is a test.")
        self.assertEqual(document.metadata["reference"], url)
        self.assertEqual(document.metadata["status_code"], 200)
        self.assertEqual(document.metadata["title"], "Example Domain")

    @patch.dict(os.environ, {"OUTCRAWL_API_KEY": "fake-key"})
    @patch("requests.post")
    def test_crawl_url_without_markdown(self, mock_post):
        """Test that a response without markdown gives an empty document."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"url": "https://example.com/", "statusCode": 200}
        mock_post.return_value = mock_response

        documents = OutcrawlCrawler().crawl_url("https://example.com")

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "")
        self.assertEqual(documents[0].metadata["reference"], "https://example.com")

    @patch.dict(os.environ, {"OUTCRAWL_API_KEY": "fake-key"})
    @patch("requests.post")
    def test_crawl_url_http_error(self, mock_post):
        """Test that an HTTP error from the API is raised."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            OutcrawlCrawler().crawl_url("https://example.com")


if __name__ == "__main__":
    unittest.main()
