# LLM Configuration

DeepSearcher supports various Large Language Models (LLMs) for processing queries and generating responses.

## 📝 Basic Configuration

```python
config.set_provider_config("llm", "(LLMName)", "(Arguments dict)")
```

## 📋 Available LLM Providers

| Provider | Description | Key Models |
|----------|-------------|------------|
| **OpenAI** | OpenAI's API for GPT models | o1-mini, GPT-4 |
| **DeepSeek** | DeepSeek AI offering | deepseek-reasoner, coder |
| **Anthropic** | Anthropic's Claude models | claude-sonnet-4-0 |
| **Gemini** | Google's Gemini models | gemini-1.5-pro, gemini-2.0-flash |
| **XAI** | X.AI's Grok models | grok-2-latest |
| **Ollama** | Local LLM deployment | llama3, qwq, etc. |
| **SiliconFlow** | Enterprise AI platform | deepseek-r1 |
| **TogetherAI** | Multiple model options | llama-4, deepseek |
| **PPIO** | Cloud AI infrastructure | deepseek, llama |
| **Volcengine** | ByteDance LLM platform | deepseek-r1 |
| **GLM** | ChatGLM models | glm-4-plus |
| **Bedrock** | Amazon Bedrock LLMs | anthropic.claude, ai21.j2 |
| **Novita** | Novita AI models | Various options |
| **IBM watsonx.ai** | IBM Enterprise AI platform | Various options |
| **JiekouAI** | JiekouAI models | Claude, OpenAI, Deepseek, Grok, Qwen, etc. |

## 🔍 Provider Examples

### OpenAI

```python
config.set_provider_config("llm", "OpenAI", {"model": "o1-mini"})
```
*Requires `OPENAI_API_KEY` environment variable*

### DeepSeek

```python
config.set_provider_config("llm", "DeepSeek", {"model": "deepseek-reasoner"})
```
*Requires `DEEPSEEK_API_KEY` environment variable*

### IBM WatsonX

```python
config.set_provider_config("llm", "WatsonX", {"model": "ibm/granite-3-3-8b-instruct"})
```
*Requires `WATSONX_APIKEY`, `WATSONX_URL`, and `WATSONX_PROJECT_ID` environment variables*

## 📚 Additional Providers

??? example "DeepSeek from SiliconFlow"

    ```python
    config.set_provider_config("llm", "SiliconFlow", {"model": "deepseek-ai/DeepSeek-R1"})
    ```
    *Requires `SILICONFLOW_API_KEY` environment variable*
    
    More details about SiliconFlow: [https://docs.siliconflow.cn/quickstart](https://docs.siliconflow.cn/quickstart)

??? example "DeepSeek from TogetherAI"

    *Requires `TOGETHER_API_KEY` environment variable and `pip install together`*
    
    For DeepSeek R1:
    ```python
    config.set_provider_config("llm", "TogetherAI", {"model": "deepseek-ai/DeepSeek-R1"})
    ```
    
    For Llama 4:
    ```python
    config.set_provider_config("llm", "TogetherAI", {"model": "meta-llama/Llama-4-Scout-17B-16E-Instruct"})
    ```
    
    More details about TogetherAI: [https://www.together.ai/](https://www.together.ai/)

??? example "XAI Grok"

    ```python
    config.set_provider_config("llm", "XAI", {"model": "grok-2-latest"})
    ```
    *Requires `XAI_API_KEY` environment variable*
    
    More details about XAI Grok: [https://docs.x.ai/docs/overview#featured-models](https://docs.x.ai/docs/overview#featured-models)

??? example "Claude"

    ```python
    config.set_provider_config("llm", "Anthropic", {"model": "claude-sonnet-4-0"})
    ```
    *Requires `ANTHROPIC_API_KEY` environment variable*
    
    More details about Anthropic Claude: [https://docs.anthropic.com/en/home](https://docs.anthropic.com/en/home)

??? example "Google Gemini"

    ```python
    config.set_provider_config('llm', 'Gemini', { 'model': 'gemini-2.0-flash' })
    ```
    *Requires `GEMINI_API_KEY` environment variable and `pip install google-genai`*
    
    More details about Gemini: [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)

??? example "DeepSeek from PPIO"

    ```python
    config.set_provider_config("llm", "PPIO", {"model": "deepseek/deepseek-r1-turbo"})
    ```
    *Requires `PPIO_API_KEY` environment variable*
    
    More details about PPIO: [https://ppinfra.com/docs/get-started/quickstart.html](https://ppinfra.com/docs/get-started/quickstart.html)

??? example "Ollama"

    ```python
    config.set_provider_config("llm", "Ollama", {"model": "qwq"})
    ```
    
    Follow [these instructions](https://github.com/jmorganca/ollama) to set up and run a local Ollama instance:
    
    1. [Download](https://ollama.ai/download) and install Ollama
    2. View available models via the [model library](https://ollama.ai/library)
    3. Pull models with `ollama pull <name-of-model>`
    4. By default, Ollama has a REST API on [http://localhost:11434](http://localhost:11434)

??? example "Volcengine"

    ```python
    config.set_provider_config("llm", "Volcengine", {"model": "deepseek-r1-250120"})
    ```
    *Requires `VOLCENGINE_API_KEY` environment variable*
    
    More details about Volcengine: [https://www.volcengine.com/docs/82379/1099455](https://www.volcengine.com/docs/82379/1099455)

??? example "GLM"

    ```python
    config.set_provider_config("llm", "GLM", {"model": "glm-4-plus"})
    ```
    *Requires `GLM_API_KEY` environment variable and `pip install zhipuai`*
    
    More details about GLM: [https://bigmodel.cn/dev/welcome](https://bigmodel.cn/dev/welcome)

??? example "Amazon Bedrock"

    ```python
    config.set_provider_config("llm", "Bedrock", {"model": "us.deepseek.r1-v1:0"})
    ```
    *Requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables and `pip install boto3`*
    
    More details about Amazon Bedrock: [https://docs.aws.amazon.com/bedrock/](https://docs.aws.amazon.com/bedrock/)

??? example "Aliyun Bailian"

    ```python
    config.set_provider_config("llm", "OpenAI", {"model": "deepseek-r1", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"})
    ```
    *Requires `OPENAI_API_KEY` environment variable*
    
    More details about Aliyun Bailian models: [https://bailian.console.aliyun.com](https://bailian.console.aliyun.com) 

??? example "IBM watsonx.ai LLM"

    ```python
    config.set_provider_config("llm", "WatsonX", {"model": "ibm/granite-3-3-8b-instruct"})
    ```

    With custom parameters:
    ```python
    config.set_provider_config("llm", "WatsonX", {
        "model": "ibm/granite-3-3-8b-instruct",
        "max_new_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    })
    ```

    With space_id instead of project_id:
    ```python
    config.set_provider_config("llm", "WatsonX", {
        "model": "ibm/granite-3-3-8b-instruct""
    })
    ```

    *Requires `WATSONX_APIKEY`, `WATSONX_URL`, and `WATSONX_PROJECT_ID` environment variables and `pip install ibm-watsonx-ai`*

    More details about WatsonX: [https://www.ibm.com/products/watsonx-ai/foundation-models](https://www.ibm.com/products/watsonx-ai/foundation-models)
```

??? example "JiekouAI"

    ```python
    config.set_provider_config("llm", "JiekouAI", {"model": "claude-sonnet-4-5-20250929"})
    ```
    *Requires `JIEKOU_API_KEY` environment variable*
    
    More details about JiekouAI: [https://docs.jiekou.ai/docs/support/quickstart](https://docs.jiekou.ai/docs/support/quickstart)

