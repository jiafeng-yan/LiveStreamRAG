import os
import json
import requests
import base64
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from io import BytesIO
import redis
import hashlib
import time
from sentence_transformers import SentenceTransformer, util
from config import APP_CONFIG
import asyncio
import aiohttp

# 新增导入: 用于本地HF模型
try:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
    print('本地模型正常载入。')
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(e)
    print("警告: `transformers` 或 `torch` 未安装或报错。本地模型功能将不可用。")
    print("请运行: pip install -U torch transformers")


class OCRProcessor:
    def __init__(self, use_local_model: bool = True, use_redis=False, use_semantic=True, similarity_threshold=0.85):
        """
        初始化OCR处理器
        
        Args:
            use_local_model: 是否使用本地HF模型进行OCR。若为True，则使用本地模型；否则使用API。
            use_redis: 是否使用Redis进行去重
            use_semantic: 是否使用语义相似度进行去重
            similarity_threshold: 语义相似度阈值
        """
        # --- 新增: 控制模型选择 ---
        self.use_local_model = use_local_model

        self.use_redis = use_redis
        self.use_semantic = use_semantic
        self.similarity_threshold = similarity_threshold
        self.processed_comments = set()  # 存储已处理的评论
        
        self.perf_config = APP_CONFIG["ocr"]["performance_optimization"]
        self.min_text_diff = self.perf_config.get("min_text_diff", 30)
        
        self.vlm_models = list(APP_CONFIG["ocr"]["vlm_models"].keys())
        self.current_model_index = 0
        self.max_retries = APP_CONFIG["ocr"]["max_retries"]
        self.model_configs = APP_CONFIG["ocr"]["vlm_models"]
        
        self.redis_client = None
        if self.use_redis:
            self._initialize_redis()
        
        self.embedding_model = None
        if self.use_semantic:
            self._initialize_semantic_model()
            
        self.max_local_cache = APP_CONFIG["deduplication"]["local_cache_size"]
        self.last_text_content = ""
        
        # --- API 配置 ---
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = APP_CONFIG["ocr"]["api_url"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/yourproject", # 请替换为您的项目地址
        }

        # --- 新增: 本地模型初始化 ---
        self.local_model = None
        self.local_processor = None
        self.device = None
        if self.use_local_model:
            self._initialize_local_model()

    def _initialize_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.Redis(
                host=APP_CONFIG["redis"]["host"],
                port=APP_CONFIG["redis"]["port"],
                db=APP_CONFIG["redis"]["db"],
                password=APP_CONFIG["redis"]["password"],
                decode_responses=True
            )
            self.redis_client.ping()
            print("Redis连接成功")
            self._cleanup_old_redis_entries()
        except Exception as e:
            print(f"Redis连接失败: {e}")
            self.use_redis = False

    def _cleanup_old_redis_entries(self):
        """如果Redis中的缓存超过上限，则删除最旧的条目"""
        redis_prefix = APP_CONFIG["redis"]["prefix"]
        redis_keys = self.redis_client.keys(f"{redis_prefix}*")
        max_items = APP_CONFIG["deduplication"]["max_cache_items"]
        if len(redis_keys) > max_items:
            oldest_keys = sorted(redis_keys, 
                                key=lambda k: float(self.redis_client.hget(k, "timestamp") or 0))
            to_delete = oldest_keys[:len(redis_keys) - max_items]
            if to_delete:
                self.redis_client.delete(*to_delete)
                print(f"已从Redis中删除 {len(to_delete)} 条最旧的评论")

    def _initialize_semantic_model(self):
        """初始化语义模型"""
        try:
            from knowledge_base.embedding import EmbeddingModel
            model_name = APP_CONFIG["semantic"]["model"]
            self.embedding_model = EmbeddingModel(model_name)
            print(f"语义模型 {model_name} 加载成功")
        except ImportError:
            print("无法导入EmbeddingModel模块，将禁用语义相似度功能")
            self.use_semantic = False
        except Exception as e:
            print(f"语义模型加载失败: {e}")
            self.use_semantic = False
    
    def _initialize_local_model(self):
        """初始化本地Hugging Face模型"""
        if not TRANSFORMERS_AVAILABLE:
            print("错误: Transformers库未安装，无法初始化本地模型。")
            self.use_local_model = False
            return

        try:
            config = APP_CONFIG["ocr"]["local_hf_model"]
            model_name = config["name"]
            self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"正在从 {model_name} 加载本地模型到 {self.device}...")
            
            # 加载处理器
            self.local_processor = AutoProcessor.from_pretrained(model_name)
            
            # 加载模型
            self.local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

            print("本地模型加载成功。")

        except Exception as e:
            print(f"加载本地模型失败: {e}")
            self.use_local_model = False
            self.local_model = None
            self.local_processor = None
    
    async def process_image(self, image_path):
        """
        处理图像提取评论。
        根据初始化参数 `use_local_model` 自动选择OCR引擎。
        """
        # 图像预处理
        try:
            img = Image.open(image_path)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)
            img.save(image_path)
        except Exception as e:
            print(f"图像预处理失败: {e}")
        
        # 根据配置选择调用本地模型或VLM API
        if self.use_local_model:
            comments_text = await self._process_image_with_local_model(image_path)
        else:
            comments_text = await self.process_image_with_vlm(image_path)

        # 评论内容相似性检测
        if len(comments_text) < self.min_text_diff and self.last_text_content:
            text_diff = len(set(comments_text) - set(self.last_text_content))
            if text_diff < self.min_text_diff:
                print(f"图像内容变化较小 (差异: {text_diff} < {self.min_text_diff})，跳过处理")
                return []
                
        self.last_text_content = comments_text
        
        comments = []
        if comments_text:
            for line in comments_text.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("问题:") and not line.startswith("回答:"):
                    line = self.clean_comment(line)
                    if line:
                        comments.append(line)
                        
        return self.deduplicate_comments(comments)

    async def _process_image_with_local_model(self, image_path: str) -> str:
        """
        (新增) 使用本地HF模型处理图像提取评论。
        """
        if not self.local_model or not self.local_processor:
            print("本地模型未初始化，无法处理图像。")
            return ""

        def _inference():
            """在同步函数中执行模型推理，以便于在线程中运行。"""
            try:
                # 1. 加载图像
                raw_image = Image.open(image_path).convert('RGB')

                # 2. 获取 Prompt 和配置
                config = APP_CONFIG["ocr"]["local_hf_model"]
                prompt_text = config.get("prompt", "从这张图片中提取所有的评论文本。")
                max_tokens = config.get("max_tokens", 1000)
                
                # 构建适用于Llava模型的prompt
                prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"

                # 3. 处理输入
                inputs = self.local_processor(prompt, raw_image, return_tensors='pt').to(self.device)

                # 4. 生成文本
                print("正在使用本地模型进行推理...")
                output = self.local_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

                # 5. 解码输出并提取有效内容
                decoded_output = self.local_processor.decode(output[0], skip_special_tokens=True)
                response_part = decoded_output.split("ASSISTANT:")[-1].strip()
                print("本地模型推理完成。")
                return response_part

            except Exception as e:
                print(f"使用本地模型处理图像时发生错误: {e}")
                return ""

        # 在单独的线程中运行同步的推理代码，避免阻塞asyncio事件循环
        return await asyncio.to_thread(_inference)

    def get_current_model(self):
        """获取当前使用的VLM模型"""
        return self.vlm_models[self.current_model_index]
        
    def switch_to_next_model(self):
        """切换到下一个VLM模型"""
        self.current_model_index = (self.current_model_index + 1) % len(self.vlm_models)
        print(f"已切换到下一个VLM模型: {self.get_current_model()}")
        
    def get_model_payload(self, image_path, current_model):
        """根据当前模型生成请求负载"""
        model_config = self.model_configs[current_model]
        prompt = model_config["prompt"]
        format_type = model_config["format"]
        max_tokens = model_config.get("max_tokens", 1000)
        
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            
        if format_type == "qwen":
            content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]
        elif format_type == "gemini":
            content = [{"type": "text", "text": prompt}, {"type": "image", "image": {"data": image_base64}}]
        else:
            content = [{"type": "text", "text": prompt}, {"type": "image", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]
            
        return {
            "model": current_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": APP_CONFIG["ocr"]["temperature"]
        }
    
    async def process_image_with_vlm(self, image_path):
        """使用VLM API处理图像提取评论"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            current_model = self.get_current_model()
            payload = self.get_model_payload(image_path, current_model)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, headers=self.headers, json=payload, timeout=30) as response:
                        if response.status == 429:
                            print(f"模型 {current_model} 已达到使用限制，切换到下一个模型...")
                            self.switch_to_next_model()
                            retry_count += 1
                            continue
                            
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"API请求失败: {response.status}, {error_text}")
                            retry_count += 1
                            continue
                            
                        result = await response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        if not content.strip():
                            print(f"模型 {current_model} 返回了空内容，切换到下一个模型...")
                            self.switch_to_next_model()
                            retry_count += 1
                            continue
                            
                        return content
                        
            except Exception as e:
                print(f"处理图像时发生错误: {e}")
                retry_count += 1
                
        print(f"已达到最大重试次数 ({self.max_retries})，无法处理图像")
        return ""
        
    def clean_comment(self, comment):
        """清理评论文本，去除无关内容"""
        import re
        comment = re.sub(r"^\d+[\.\)\]]\s*", "", comment)
        comment = re.sub(r"^(问题|回答|Question|Answer)[：:]\s*", "", comment)
        return comment.strip()
        
    def deduplicate_comments(self, comments):
        """去除重复评论"""
        if not comments:
            return []
            
        unique_comments = []
        for comment in comments:
            if not comment.strip():
                continue
            if comment in self.processed_comments:
                continue
                
            if self.use_redis and self.redis_client:
                redis_prefix = APP_CONFIG["redis"]["prefix"]
                comment_key = f"{redis_prefix}{comment}"
                if self.redis_client.exists(comment_key):
                    continue
                self.redis_client.hset(comment_key, mapping={"comment": comment, "timestamp": time.time()})
                self.redis_client.expire(comment_key, APP_CONFIG["redis"]["ttl"])
                
            if self.use_semantic and self.embedding_model:
                try:
                    is_similar = False
                    comment_embedding = self.embedding_model.get_embeddings([comment])[0]
                    for processed in self.processed_comments:
                        processed_embedding = self.embedding_model.get_embeddings([processed])[0]
                        similarity = self.embedding_model.compute_similarity(comment_embedding, processed_embedding)
                        if similarity > self.similarity_threshold:
                            is_similar = True
                            print(f"发现语义相似评论: '{comment}' 与 '{processed}', 相似度: {similarity:.4f}")
                            break
                    if is_similar:
                        continue
                except Exception as e:
                    print(f"计算语义相似度时出错: {e}")
            
            unique_comments.append(comment)
            self.processed_comments.add(comment)
            
            if len(self.processed_comments) > self.max_local_cache:
                self.processed_comments = set(list(self.processed_comments)[1:])
                
        return unique_comments

