# Embedding Model Configuration

DeepSearcher supports various embedding models to convert text into vector representations for semantic search.

## 📝 Basic Configuration

```python
config.set_provider_config("embedding", "(EmbeddingModelName)", "(Arguments dict)")
```

## 📋 Available Embedding Providers

| Provider | Description | Key Features |
|----------|-------------|--------------|
| **OpenAIEmbedding** | OpenAI's text embedding models | High quality, production-ready |
| **MilvusEmbedding** | Built-in embedding models via Pymilvus | Multiple model options |
| **VoyageEmbedding** | VoyageAI embedding models | Specialized for search |
| **BedrockEmbedding** | Amazon Bedrock embedding | AWS integration |
| **GeminiEmbedding** | Google's Gemini embedding | High performance |
| **GLMEmbedding** | ChatGLM embeddings | Chinese language support |
| **OllamaEmbedding** | Local embedding with Ollama | Self-hosted option |
| **PPIOEmbedding** | PPIO cloud embedding | Scalable solution |
| **SiliconflowEmbedding** | Siliconflow's models | Enterprise support |
| **VolcengineEmbedding** | Volcengine embedding | High throughput |
| **NovitaEmbedding** | Novita AI embedding | Cost-effective |
| **SentenceTransformerEmbedding** | Sentence Transfomer Embedding | Self-hosted option |
| **IBM watsonx.ai** | Various options | IBM's Enterprise AI platform |
| **JiekouAIEmbedding** | JiekouAI embedding | High quality, cost-effective |

## 🔍 Provider Examples

### OpenAI Embedding

```python
config.set_provider_config("embedding", "OpenAIEmbedding", {"model": "text-embedding-3-small"})
```
*Requires `OPENAI_API_KEY` environment variable*

### Milvus Built-in Embedding

```python
config.set_provider_config("embedding", "MilvusEmbedding", {"model": "BAAI/bge-base-en-v1.5"})
```

```python
config.set_provider_config("embedding", "MilvusEmbedding", {"model": "jina-embeddings-v3"})
```
*For Jina's embedding model, requires `JINAAI_API_KEY` environment variable*

### VoyageAI Embedding

```python
config.set_provider_config("embedding", "VoyageEmbedding", {"model": "voyage-3"})
```
*Requires `VOYAGE_API_KEY` environment variable and `pip install voyageai`*

## 📚 Additional Providers

??? example "Amazon Bedrock"

    ```python
    config.set_provider_config("embedding", "BedrockEmbedding", {"model": "amazon.titan-embed-text-v2:0"})
    ```
    *Requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables and `pip install boto3`*

??? example "Novita AI"

    ```python
    config.set_provider_config("embedding", "NovitaEmbedding", {"model": "baai/bge-m3"})
    ```
    *Requires `NOVITA_API_KEY` environment variable*

??? example "Siliconflow"

    ```python
    config.set_provider_config("embedding", "SiliconflowEmbedding", {"model": "BAAI/bge-m3"})
    ```
    *Requires `SILICONFLOW_API_KEY` environment variable*

??? example "Volcengine"

    ```python
    config.set_provider_config("embedding", "VolcengineEmbedding", {"model": "doubao-embedding-text-240515"})
    ```
    *Requires `VOLCENGINE_API_KEY` environment variable*

??? example "GLM"

    ```python
    config.set_provider_config("embedding", "GLMEmbedding", {"model": "embedding-3"})
    ```
    *Requires `GLM_API_KEY` environment variable and `pip install zhipuai`*

??? example "Google Gemini"

    ```python
    config.set_provider_config("embedding", "GeminiEmbedding", {"model": "text-embedding-004"})
    ```
    *Requires `GEMINI_API_KEY` environment variable and `pip install google-genai`*

??? example "Ollama"

    ```python
    config.set_provider_config("embedding", "OllamaEmbedding", {"model": "bge-m3"})
    ```
    *Requires local Ollama installation and `pip install ollama`*

??? example "PPIO"

    ```python
    config.set_provider_config("embedding", "PPIOEmbedding", {"model": "baai/bge-m3"})
    ```
    *Requires `PPIO_API_KEY` environment variable*

??? example "SentenceTransformer"

    ```python
    config.set_provider_config("embedding", "SentenceTransformerEmbedding", {"model": "BAAI/bge-large-zh-v1.5"})
    ```
    *Requires `pip install sentence-transformers`*

??? example "IBM WatsonX"

    ```python
    config.set_provider_config("embedding", "WatsonXEmbedding", {"model": "ibm/slate-125m-english-rtrvr-v2"})
    ```
    *Requires `pip install ibm-watsonx-ai`*

??? example "JiekouAI"

    ```python
    config.set_provider_config("embedding", "JiekouAIEmbedding", {"model": "qwen/qwen3-embedding-8b"})
    ```
    *Requires `JIEKOU_API_KEY` environment variable*
