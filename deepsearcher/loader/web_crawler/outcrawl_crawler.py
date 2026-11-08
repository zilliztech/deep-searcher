import os
from typing import List

import requests
from langchain_core.documents import Document

from deepsearcher.loader.web_crawler.base import BaseCrawler

OUTCRAWL_SCRAPE_URL = "https://api.outcrawl.ai/v1/scrape"


class OutcrawlCrawler(BaseCrawler):
    """
    Web crawler using the Outcrawl scrape API.

    This crawler loads a page in a cloud browser through Outcrawl and returns its
    content as markdown for further processing, including pages that only render
    with JavaScript.
    """

    def __init__(self, **kwargs):
        """
        Initialize the OutcrawlCrawler.

        Args:
            **kwargs: Optional keyword arguments.

        Raises:
            ValueError: If the OUTCRAWL_API_KEY environment variable is not set.
        """
        super().__init__(**kwargs)
        self.outcrawl_api_key = os.getenv("OUTCRAWL_API_KEY")
        if not self.outcrawl_api_key:
            raise ValueError("Missing OUTCRAWL_API_KEY environment variable")

    def crawl_url(self, url: str) -> List[Document]:
        """
        Scrape a single URL using the Outcrawl scrape API.

        Args:
            url: The URL to crawl.

        Returns:
            A list containing a single Document object with the markdown content and metadata.

        Raises:
            HTTPError: If the request to Outcrawl fails.
        """
        response = requests.post(
            OUTCRAWL_SCRAPE_URL,
            headers={
                "Authorization": f"Bearer {self.outcrawl_api_key}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown"]},
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        metadata = {
            "reference": url,
            "status_code": data.get("statusCode"),
            **(data.get("metadata") or {}),
        }

        return [Document(page_content=data.get("markdown") or "", metadata=metadata)]
