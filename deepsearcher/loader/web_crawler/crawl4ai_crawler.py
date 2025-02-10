import asyncio
from typing import List
from langchain_core.documents import Document
from deepsearcher.loader.web_crawler.base import BaseCrawler
from crawl4ai import AsyncWebCrawler


class Crawl4AICrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crawler = None  # Lazy init

    def _lazy_init(self):
        if self.crawler is None:
            self.crawler = AsyncWebCrawler()

    async def _async_crawl(self, url: str) -> Document:
        if self.crawler is None:
            self._lazy_init()

        async with self.crawler as crawler:
            result = await crawler.arun(url)

            markdown_content = result.markdown or ""

            metadata = {
                "reference": url,
                "success": result.success,
                "status_code": result.status_code,
                "media": result.media,
                "links": result.links
            }

            if hasattr(result, "metadata") and result.metadata:
                metadata["title"] = result.metadata.get("title", "")
                metadata["author"] = result.metadata.get("author", "")

            return Document(page_content=markdown_content, metadata=metadata)

    def crawl_url(self, url: str) -> List[Document]:
        try:
            document = asyncio.run(self._async_crawl(url))
            return [document]
        except Exception as e:
            print(f"Error during crawling {url}: {e}")
            return []


crawler = Crawl4AICrawler()
res = crawler.crawl_url("https://lilianweng.github.io/posts/2023-06-23-agent/")
print(res[0].metadata['title'])
print(res[0].metadata['reference'])
