import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import aiohttp
import json
import os
from typing import Dict, Any, Optional, List

class ResponseLLMClient:
    """
    一个统一的大语言模型客户端，
    支持调用本地Hugging Face模型或通过API调用云端模型。
    """

    def __init__(
        self,
        use_local_model: bool,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "deepseek/deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 500,
        torch_dtype: Any = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        """
        初始化客户端。

        Args:
            use_local_model (bool): True表示使用本地模型，False表示使用API。
            model_path (Optional[str]): 本地模型的路径 (当 use_local_model=True 时必需).
            api_key (Optional[str]): API密钥 (当 use_local_model=False 时必需).
            model (str): API调用的模型名称.
            temperature (float): 控制生成文本的随机性.
            max_tokens (int): 最大生成token数.
            torch_dtype (Any): 本地模型加载的数据类型.
        """
        self.use_local_model = use_local_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.use_local_model:
            # --- 本地模型初始化逻辑 ---
            if not model_path:
                raise ValueError("使用本地模型时，必须提供 'model_path'")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = model_path
            
            print(f"模式: 本地模型。正在从 '{self.model_path}' 加载模型到 {self.device}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=self.device
            )
            print("本地模型加载完成。")

        else:
            # --- API调用初始化逻辑 ---
            if not api_key:
                raise ValueError("使用API时，必须提供 'api_key'")

            self.api_key = api_key
            self.model_name = model
            self.base_url = "https://openrouter.ai/api/v1"
            print(f"模式: API调用。将使用模型 '{self.model_name}'。")


    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        根据初始化配置，调用相应模型生成文本。

        Args:
            prompt (str): 用户提示词.
            system_prompt (Optional[str]): 系统提示词.
            temperature (Optional[float]): 温度参数，如果为None则使用初始化时设置的值.
            max_tokens (Optional[int]): 最大生成token数，如果为None则使用初始化时设置的值.
            stream (bool): 是否流式输出.

        Returns:
            str: 生成的完整文本.
        """
        if self.use_local_model:
            return self._generate_local(prompt, system_prompt, temperature, max_tokens, stream)
        else:
            return await self._generate_api(prompt, system_prompt, temperature, max_tokens, stream)

    def _generate_local(
        self, prompt, system_prompt, temperature, max_tokens, stream
    ) -> str:
        """处理本地模型生成"""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            input_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_prompt = (system_prompt or "") + "\n" + prompt

        inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": tokens,
            "temperature": temp,
            "do_sample": temp > 0,
        }

        if stream:
            print("本地流式输出: ", end="", flush=True)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            generation_kwargs["streamer"] = streamer
        
        output_ids = self.model.generate(**generation_kwargs)
        if stream: print() # 流式结束后换行
        
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return generated_text.strip()

    async def _generate_api(
        self, prompt, system_prompt, temperature, max_tokens, stream
    ) -> str:
        """处理API调用生成"""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API请求失败: {response.status}, {error_text}")

                if stream:
                    if stream: print("API流式输出: ", end="", flush=True)
                    return await self._handle_stream_response(response)
                else:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

    async def _handle_stream_response(self, response) -> str:
        """处理API的流式响应"""
        full_text = ""
        async for line in response.content:
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        data = json.loads(line[6:])
                        content = data["choices"][0]["delta"].get("content", "")
                        full_text += content
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue # 忽略空行或其他非json行
        print() # 流式结束后换行
        return full_text

# --- 使用示例 ---
import asyncio

async def main():
    # --- 场景1: 使用本地模型 ---
    print("="*30)
    print("场景1: 初始化本地模型客户端")
    print("="*30)
    try:
        # 请将此路径替换为你的实际模型路径
        local_client = ResponseLLMClient(
            use_local_model=True,
            model_path="path/to/your/local/hf/model" 
        )
        
        system_p = "你是一个能干的助手。"
        user_p = "请给我写一首关于夏夜星空的短诗。"
        
        print("\n--- 本地模型非流式生成 ---")
        response_local = await local_client.generate(prompt=user_p, system_prompt=system_p)
        print(f"\n模型输出:\n{response_local}")
        
        print("\n--- 本地模型流式生成 ---")
        await local_client.generate(prompt=user_p, system_prompt=system_p, stream=True)

    except (ValueError, OSError) as e:
        print(f"\n运行本地模型示例时出错: {e}")
        print("请确保 'model_path' 指向一个有效的Hugging Face模型目录。")

    # --- 场景2: 使用API ---
    print("\n" + "="*30)
    print("场景2: 初始化API客户端")
    print("="*30)
    try:
        # 从环境变量或安全的地方获取API密钥
        api_key = os.environ.get("OPENROUTER_API_KEY") 
        if not api_key:
            raise ValueError("请设置环境变量 'OPENROUTER_API_KEY'")

        api_client = ResponseLLMClient(
            use_local_model=False,
            api_key=api_key,
            model="deepseek/deepseek-chat" # 可以换成OpenRouter支持的任何模型
        )

        system_p = "你是一个专业的直播评论分析师。"
        user_p = "这件衣服会不会缩水？主播身高体重多少？今天还有优惠吗？感觉有点贵。能再便宜点吗？"
        
        print("\n--- API非流式生成 ---")
        response_api = await api_client.generate(prompt=user_p, system_prompt=system_p)
        print(f"\n模型输出:\n{response_api}")

        print("\n--- API流式生成 ---")
        await api_client.generate(prompt=user_p, system_prompt=system_p, stream=True)

    except ValueError as e:
        print(f"\n运行API示例时出错: {e}")

if __name__ == '__main__':
    # 在Jupyter Notebook等已有事件循环的环境中，可以直接 await main()
    # 在普通的.py文件中，使用 asyncio.run() 来启动
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # 如果在已经有事件循环的环境（如Jupyter）中运行此脚本
        # 会抛出 RuntimeError，此时可以直接await
        print("检测到已有事件循环，请在Jupyter等环境中直接 'await main()'。")

