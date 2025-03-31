import aiohttp
import json
import os
from typing import Dict, Any, Optional, List


class OpenRouterClient:
    """Openrouter API客户端"""

    def __init__(self, api_key: str, model: str = "deepseek/deepseek-chat"):
        """
        初始化OpenRouter客户端

        Args:
            api_key: API密钥
            model: 模型名称，默认为deepseek-chat
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
        stream: bool = False,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式输出

        Returns:
            生成的文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions", headers=headers, json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败: {response.status}, {error_text}")

                if stream:
                    return self._handle_stream_response(response)
                else:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

    async def _handle_stream_response(self, response) -> str:
        """处理流式响应"""
        full_text = ""
        async for line in response.content:
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    content = data["choices"][0]["delta"].get("content", "")
                    full_text += content
                    # 实时输出（在实际应用中可以通过回调函数实现）
                    print(content, end="", flush=True)
        return full_text
