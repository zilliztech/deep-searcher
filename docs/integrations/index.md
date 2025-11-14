# Module Support

DeepSearcher supports various integration modules including embedding models, large language models, document loaders and vector databases.

## 📊 Overview

| Module Type | Count | Description |
|-------------|-------|-------------|
| [Embedding Models](#embedding-models) | 7+ | Text vectorization tools |
| [Large Language Models](#llm-support) | 11+ | Query processing and text generation |
| [Document Loaders](#document-loader) | 5+ | Parse and process documents in various formats |
| [Vector Databases](#vector-database-support) | 2+ | Store and retrieve vector data |

## 🔢 Embedding Models {#embedding-models}

Support for various embedding models to convert text into vector representations for semantic search.

| Provider | Required Environment Variables | Features |
|----------|--------------------------------|---------|
| **[Open-source models](https://milvus.io/docs/embeddings.md)** | None | Locally runnable open-source models |
| **[OpenAI](https://platform.openai.com/docs/guides/embeddings/use-cases)** | `OPENAI_API_KEY` | High-quality embeddings, easy to use |
| **[VoyageAI](https://docs.voyageai.com/embeddings/)** | `VOYAGE_API_KEY` | Embeddings optimized for retrieval |
| **[Amazon Bedrock](https://docs.aws.amazon.com/bedrock/)** | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | AWS integration, enterprise-grade |
| **[FastEmbed](https://qdrant.github.io/fastembed/)** | None | Fast lightweight embeddings |
| **[PPIO](https://ppinfra.com/model-api/product/llm-api)** | `PPIO_API_KEY` | Flexible cloud embeddings |
| **[Novita AI](https://novita.ai/docs/api-reference/model-apis-llm-create-embeddings)** | `NOVITA_API_KEY` | Rich model selection |
| **[IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai/foundation-models#ibmembedding)** | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID` | IBM's Enterprise AI platform |
| **[JiekouAI](https://jiekou.ai/?utm_source=github_deep-searcher)** | `JIEKOU_API_KEY` | JiekouAI's embedding models |

## 🧠 Large Language Models {#llm-support}

Support for various large language models (LLMs) to process queries and generate responses.

| Provider | Required Environment Variables | Features |
|----------|--------------------------------|---------|
| **[OpenAI](https://platform.openai.com/docs/models)** | `OPENAI_API_KEY` | GPT model family |
| **[DeepSeek](https://api-docs.deepseek.com/)** | `DEEPSEEK_API_KEY` | Powerful reasoning capabilities |
| **[XAI Grok](https://x.ai/blog/grok-3)** | `XAI_API_KEY` | Real-time knowledge and humor |
| **[Anthropic Claude](https://docs.anthropic.com/en/home)** | `ANTHROPIC_API_KEY` | Excellent long-context understanding |
| **[SiliconFlow](https://docs.siliconflow.cn/en/userguide/introduction)** | `SILICONFLOW_API_KEY` | Enterprise inference service |
| **[PPIO](https://ppinfra.com/model-api/product/llm-api)** | `PPIO_API_KEY` | Diverse model support |
| **[TogetherAI](https://docs.together.ai/docs/introduction)** | `TOGETHER_API_KEY` | Wide range of open-source models |
| **[Google Gemini](https://ai.google.dev/gemini-api/docs)** | `GEMINI_API_KEY` | Google's multimodal models |
| **[SambaNova](https://docs.together.ai/docs/introduction)** | `SAMBANOVA_API_KEY` | High-performance AI platform |
| **[Ollama](https://ollama.com/)** | None | Local LLM deployment |
| **[Novita AI](https://novita.ai/docs/guides/introduction)** | `NOVITA_API_KEY` | Diverse AI services |
| **[IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai/foundation-models#ibmfm)** | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID` | IBM's Enterprise AI platform |
| **[JiekouAI](https://jiekou.ai/?utm_source=github_deep-searcher)** | `JIEKOU_API_KEY` | Various open-source and closed-source models |

## 📄 Document Loader {#document-loader}

Support for loading and processing documents from various sources.

### Local File Loaders

| Loader | Supported Formats | Required Environment Variables |
|--------|-------------------|--------------------------------|
| **Built-in Loader** | PDF, TXT, MD | None |
| **[Unstructured](https://unstructured.io/)** | Multiple document formats | `UNSTRUCTURED_API_KEY`, `UNSTRUCTURED_URL` (optional) |

### Web Crawlers

| Crawler | Description | Required Environment Variables/Setup |
|---------|-------------|--------------------------------------|
| **[FireCrawl](https://docs.firecrawl.dev/introduction)** | Crawler designed for AI applications | `FIRECRAWL_API_KEY` |
| **[Jina Reader](https://jina.ai/reader/)** | High-accuracy web content extraction | `JINA_API_TOKEN` |
| **[Crawl4AI](https://docs.crawl4ai.com/)** | Browser automation crawler | Run `crawl4ai-setup` for first-time use |

## 💾 Vector Database Support {#vector-database-support}

Support for various vector databases for efficient storage and retrieval of embeddings.

| Database | Description | Features |
|----------|-------------|----------|
| **[Milvus](https://milvus.io/)** | Open-source vector database | High-performance, scalable |
| **[Zilliz Cloud](https://www.zilliz.com/)** | Managed Milvus service | Fully managed, maintenance-free |
| **[Qdrant](https://qdrant.tech/)** | Vector similarity search engine | Simple, efficient | 