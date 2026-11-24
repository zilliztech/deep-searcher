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
| **Rapid-MLX (via OpenAIEmbedding)** | Local embeddings on Apple Silicon | 768 dimensions |
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
| **JiekouAIEmbedding** | Jiekou.AI embedding | High quality, cost-effective |

## 🔍 Provider Examples

### OpenAI Embedding

```python
config.set_provider_config("embedding", "OpenAIEmbedding", {"model": "text-embedding-3-small"})
```
*Requires `OPENAI_API_KEY` environment variable*

### Rapid-MLX

Use the existing `OpenAIEmbedding` class with Rapid-MLX's `/v1/embeddings` endpoint.
Embeddings need Rapid-MLX's optional embeddings runtime. Install it in a
Python 3.12+ environment, then start a server with both models:

```shell
pip install "rapid-mlx[embeddings]"
rapid-mlx serve qwen3.5-4b-4bit --port 8000 --embedding-model mlx-community/embeddinggemma-2-4bit
```

Set `dimension` explicitly to **768** for this embedding model; the class otherwise
defaults unknown models to 1536. The server needs no API key, but the SDK requires
a non-empty placeholder. Both providers use the same base URL, including `/v1`.
If port 8000 is occupied, choose another port and update both URLs below.

Run this example in a fresh directory to load a local text file into Milvus Lite
and answer a question using the retrieved text:

```python
from pathlib import Path

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import naive_rag_query

config = Configuration()
config.set_provider_config("llm", "OpenAI", {
    "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "base_url": "http://localhost:8000/v1",
    "api_key": "not-needed",
})
config.set_provider_config("embedding", "OpenAIEmbedding", {
    "model": "mlx-community/embeddinggemma-2-4bit",
    "base_url": "http://localhost:8000/v1",
    "api_key": "not-needed",
    "dimension": 768,
})
config.set_provider_config("file_loader", "TextLoader", {})
config.set_provider_config("vector_db", "Milvus", {
    "uri": "./rapid_mlx.db",
    "default_collection": "rapid_mlx_demo",
})
init_config(config)

Path("rapid_mlx_notes.txt").write_text(
    "The observatory's telescope is named Aurora. It has a 42 cm aperture.",
    encoding="utf-8",
)
load_from_local_files(
    "rapid_mlx_notes.txt",
    collection_description="Observatory notes",
)
answer, sources = naive_rag_query("What is the telescope's name and aperture?")
print(answer)
print(f"Retrieved {len(sources)} source(s)")
```

The same provider arguments can be set in `deepsearcher/config.yaml`; see its
commented Rapid-MLX examples. Use a new collection when changing embedding models
or dimensions.

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

??? example "Jiekou.AI"

    ```python
    config.set_provider_config("embedding", "JiekouAIEmbedding", {"model": "qwen/qwen3-embedding-8b"})
    ```
    *Requires `JIEKOU_API_KEY` environment variable*
