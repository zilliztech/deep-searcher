# Web Crawler Configuration

DeepSearcher supports various web crawlers to collect data from websites for processing and indexing.

## 📝 Basic Configuration

```python
config.set_provider_config("web_crawler", "(WebCrawlerName)", "(Arguments dict)")
```

## 📋 Available Web Crawlers

| Crawler | Description | Key Feature |
|---------|-------------|-------------|
| **FireCrawlCrawler** | Cloud-based web crawling service | Simple API, managed service |
| **Crawl4AICrawler** | Browser automation crawler | Full JavaScript support |
| **JinaCrawler** | Content extraction service | High accuracy parsing |
| **OutcrawlCrawler** | Cloud browser scrape API | Clean markdown from JavaScript-rendered pages |
| **DoclingCrawler** | Doc processing with crawling | Multiple format support |

## 🔍 Web Crawler Options

### FireCrawl

[FireCrawl](https://docs.firecrawl.dev/introduction) is a cloud-based web crawling service designed for AI applications.

**Key features:**
- Simple API
- Managed Service
- Advanced Parsing

```python
config.set_provider_config("web_crawler", "FireCrawlCrawler", {})
```

??? tip "Setup Instructions"

    1. Sign up for FireCrawl and get an API key
    2. Set the API key as an environment variable:
       ```bash
       export FIRECRAWL_API_KEY="your_api_key"
       ```
    3. For more information, see the [FireCrawl documentation](https://docs.firecrawl.dev/introduction)

### Crawl4AI

[Crawl4AI](https://docs.crawl4ai.com/) is a Python package for web crawling with browser automation capabilities.

```python
config.set_provider_config("web_crawler", "Crawl4AICrawler", {"browser_config": {"headless": True, "verbose": True}})
```

??? tip "Setup Instructions"

    1. Install Crawl4AI:
       ```bash
       pip install crawl4ai
       ```
    2. Run the setup command:
       ```bash
       crawl4ai-setup
       ```
    3. For more information, see the [Crawl4AI documentation](https://docs.crawl4ai.com/)

### Jina Reader

[Jina Reader](https://jina.ai/reader/) is a service for extracting content from web pages with high accuracy.

```python
config.set_provider_config("web_crawler", "JinaCrawler", {})
```

??? tip "Setup Instructions"

    1. Get a Jina API key
    2. Set the API key as an environment variable:
       ```bash
       export JINA_API_TOKEN="your_api_key"
       # or
       export JINAAI_API_KEY="your_api_key"
       ```
    3. For more information, see the [Jina Reader documentation](https://jina.ai/reader/)

### Outcrawl

[Outcrawl](https://outcrawl.ai) is web infrastructure for AI agents. Its scrape API loads a page in a cloud browser and returns the content as markdown.

```python
config.set_provider_config("web_crawler", "OutcrawlCrawler", {})
```

??? tip "Setup Instructions"

    1. Get an Outcrawl API key at [outcrawl.ai](https://outcrawl.ai)
    2. Set the API key as an environment variable:
       ```bash
       export OUTCRAWL_API_KEY="your_api_key"
       ```

### Docling Crawler

[Docling](https://docling-project.github.io/docling/) provides web crawling capabilities alongside its document processing features.

```python
config.set_provider_config("web_crawler", "DoclingCrawler", {})
```

??? tip "Setup Instructions"

    1. Install Docling:
       ```bash
       pip install docling
       ```
    2. For information on supported formats, see the [Docling documentation](https://docling-project.github.io/docling/usage/supported_formats/#supported-output-formats) 