import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class GLM(BaseLLM):
    """
    https://open.bigmodel.cn/api/paas/v4/
    """

    def __init__(self, model: str = "glm-4-plus", **kwargs):
        from zhipuai import ZhipuAI

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("GLM_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("GLM_BASE_URL", default="https://open.bigmodel.cn/api/paas/v4/")
        self.client = ZhipuAI(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        if not completion.choices or completion.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
