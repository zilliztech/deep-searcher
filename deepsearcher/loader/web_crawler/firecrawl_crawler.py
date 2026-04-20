import os
from typing import List, Optional

try:
    from firecrawl import FirecrawlApp, ScrapeOptions
except ImportError:
    try:
        from firecrawl import FirecrawlApp
        from firecrawl.types import ScrapeOptions
    except ImportError:
        from firecrawl import FirecrawlApp, V1ScrapeOptions as ScrapeOptions  # type: ignore[attr-defined]
from langchain_core.documents import Document

from deepsearcher.loader.web_crawler.base import BaseCrawler


class FireCrawlCrawler(BaseCrawler):
    """
    Web crawler using the FireCrawl service.

    This crawler uses the FireCrawl service to crawl web pages and convert them
    into markdown format for further processing. It supports both single-page scraping
    and recursive crawling of multiple pages.
    """

    def __init__(self, **kwargs):
        """
        Initialize the FireCrawlCrawler.

        Args:
            **kwargs: Optional keyword arguments.
        """
        super().__init__(**kwargs)
        self.app = None

    def crawl_url(
        self,
        url: str,
        max_depth: Optional[int] = None,
        limit: Optional[int] = None,
        allow_backward_links: Optional[bool] = None,
    ) -> List[Document]:
        """
        Dynamically crawls a URL using either scrape_url or crawl_url:

        - Uses scrape_url for single-page extraction if no params are provided.
        - Uses crawl_url to recursively gather pages when any param is provided.

        Args:
            url (str): The starting URL to crawl.
            max_depth (Optional[int]): Maximum depth for recursive crawling (default: 2).
            limit (Optional[int]): Maximum number of pages to crawl (default: 20).
            allow_backward_links (Optional[bool]): Allow crawling pages outside the URL's children (default: False).

        Returns:
            List[Document]: List of Document objects with page content and metadata.
        """
        # Lazy init
        self.app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

        # if user just inputs a single url as param
        # scrape single page
        if max_depth is None and limit is None and allow_backward_links is None:
            # Call the new Firecrawl API, passing formats directly
            scrape_response = self.app.scrape_url(url=url, formats=["markdown"])
            data = scrape_response.model_dump()
            return [
                Document(
                    page_content=data.get("markdown", ""),
                    metadata={"reference": url, **data.get("metadata", {})},
                )
            ]

        # else, crawl multiple pages based on users' input params
        # set default values if not provided
        crawl_response = self.app.crawl_url(
            url=url,
            limit=limit or 20,
            max_depth=max_depth or 2,
            allow_backward_links=allow_backward_links or False,
            scrape_options=ScrapeOptions(formats=["markdown"]),
            poll_interval=5,
        )
        items = crawl_response.model_dump().get("data", [])

        documents: List[Document] = []
        for item in items:
            # Support items that are either dicts or Pydantic sub-models
            item_dict = item.model_dump() if hasattr(item, "model_dump") else item
            md = item_dict.get("markdown", "")
            meta = item_dict.get("metadata", {})
            meta["reference"] = meta.get("url", url)
            documents.append(Document(page_content=md, metadata=meta))

        return documents
