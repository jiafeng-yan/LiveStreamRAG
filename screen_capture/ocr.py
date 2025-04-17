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
from config.app_config import APP_CONFIG
import asyncio
import aiohttp

class OCRProcessor:
    def __init__(self, use_redis=False, use_semantic=True, similarity_threshold=0.85):
        """
        初始化OCR处理器
        
        Args:
            use_redis: 是否使用Redis进行去重
            use_semantic: 是否使用语义相似度进行去重
            similarity_threshold: 语义相似度阈值
        """
        self.use_redis = use_redis
        self.use_semantic = use_semantic
        self.similarity_threshold = similarity_threshold
        self.processed_comments = set()  # 存储已处理的评论
        
        # 获取配置中的性能优化参数
        self.perf_config = APP_CONFIG["ocr"]["performance_optimization"]
        self.min_text_diff = self.perf_config.get("min_text_diff", 30)
        
        # VLM模型配置
        self.vlm_models = list(APP_CONFIG["ocr"]["vlm_models"].keys())
        self.current_model_index = 0
        self.max_retries = APP_CONFIG["ocr"]["max_retries"]
        self.model_configs = APP_CONFIG["ocr"]["vlm_models"]
        
        # Redis连接（如果启用）
        self.redis_client = None
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=APP_CONFIG["redis"]["host"],
                    port=APP_CONFIG["redis"]["port"],
                    db=APP_CONFIG["redis"]["db"],
                    password=APP_CONFIG["redis"]["password"],
                    decode_responses=True
                )
                self.redis_client.ping()  # 测试连接
                print("Redis连接成功")
                
                # 如果Redis中的评论数量超过上限，则删除最旧的评论
                redis_prefix = APP_CONFIG["redis"]["prefix"]
                redis_keys = self.redis_client.keys(f"{redis_prefix}*")
                if len(redis_keys) > APP_CONFIG["deduplication"]["max_cache_items"]:
                    # 按时间戳排序并删除最旧的评论
                    oldest_keys = sorted(redis_keys, 
                                        key=lambda k: float(self.redis_client.hget(k, "timestamp") or 0))
                    to_delete = oldest_keys[:len(redis_keys) - APP_CONFIG["deduplication"]["max_cache_items"]]
                    if to_delete:
                        self.redis_client.delete(*to_delete)
                        print(f"已从Redis中删除 {len(to_delete)} 条最旧的评论")
                
            except Exception as e:
                print(f"Redis连接失败: {e}")
                self.use_redis = False
        
        # 语义模型（如果启用）
        self.embedding_model = None
        if self.use_semantic:
            try:
                # 尝试导入EmbeddingModel
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
                
        # 限制内存中的评论数量
        self.max_local_cache = APP_CONFIG["deduplication"]["local_cache_size"]
        
        # 上一次处理的图像文本内容（用于变化检测）
        self.last_text_content = ""
        
        # API 配置
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = APP_CONFIG["ocr"]["api_url"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/yourproject",
        }
    
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
        
        # 准备图像
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
        # 根据模型类型构建不同的请求格式
        if format_type == "qwen":
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        elif format_type == "gemini":
            content = [
                {"type": "text", "text": prompt},
                {"type": "image", "image": {"data": image_base64}}
            ]
        else:  # 默认为 gpt4v 格式
            content = [
                {"type": "text", "text": prompt},
                {"type": "image", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
            
        return {
            "model": current_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": APP_CONFIG["ocr"]["temperature"]
        }
    
    async def process_image_with_vlm(self, image_path):
        """使用VLM处理图像提取评论"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            current_model = self.get_current_model()
            payload = self.get_model_payload(image_path, current_model)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    ) as response:
                        # 检查响应状态
                        if response.status == 429:  # 速率限制
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
                        
                        # 检查内容是否为空
                        if not content.strip():
                            print(f"模型 {current_model} 返回了空内容，切换到下一个模型...")
                            self.switch_to_next_model()
                            retry_count += 1
                            continue
                            
                        # 成功获取内容，重置重试计数
                        retry_count = 0
                        return content
                        
            except Exception as e:
                print(f"处理图像时发生错误: {e}")
                retry_count += 1
                
        print(f"已达到最大重试次数 ({self.max_retries})，无法处理图像")
        return ""
        
    async def process_image(self, image_path):
        """处理图像提取评论"""
        # 图像预处理：增强对比度和锐度
        try:
            img = Image.open(image_path)
            # 增强对比度
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            # 增强锐度
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)
            # 保存增强后的图像
            img.save(image_path)
        except Exception as e:
            print(f"图像预处理失败: {e}")
            
        # 使用VLM提取评论 - 直接等待异步调用，不使用run_until_complete
        comments_text = await self.process_image_with_vlm(image_path)
        
        # 评论内容相似性检测
        if len(comments_text) < self.min_text_diff and self.last_text_content:
            text_diff = len(set(comments_text) - set(self.last_text_content))
            if text_diff < self.min_text_diff:
                print(f"图像内容变化较小 (差异: {text_diff} < {self.min_text_diff})，跳过处理")
                return []
                
        # 更新上次内容
        self.last_text_content = comments_text
        
        # 从文本中提取评论列表
        comments = []
        if comments_text:
            for line in comments_text.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("问题:") and not line.startswith("回答:"):
                    # 去除常见的行号和前缀
                    line = self.clean_comment(line)
                    if line:
                        comments.append(line)
                        
        # 去重处理
        filtered_comments = self.deduplicate_comments(comments)
        
        return filtered_comments
        
    def clean_comment(self, comment):
        """清理评论文本，去除无关内容"""
        # 去除行号和常见前缀
        import re
        
        # 去除数字开头，如 "1. ", "2) ", "[3]" 等
        comment = re.sub(r"^\d+[\.\)\]]\s*", "", comment)
        
        # 去除问题/回答前缀
        comment = re.sub(r"^(问题|回答|Question|Answer)[：:]\s*", "", comment)
        
        return comment.strip()
        
    def deduplicate_comments(self, comments):
        """去除重复评论"""
        if not comments:
            return []
            
        unique_comments = []
        
        for comment in comments:
            # 检查是否为空评论
            if not comment.strip():
                continue
                
            # 1. 精确匹配去重
            if comment in self.processed_comments:
                continue
                
            # 2. Redis去重（如果启用）
            if self.use_redis and self.redis_client:
                redis_prefix = APP_CONFIG["redis"]["prefix"]
                comment_key = f"{redis_prefix}{comment}"
                
                if self.redis_client.exists(comment_key):
                    continue
                    
                # 将新评论添加到Redis
                self.redis_client.hset(comment_key, mapping={
                    "comment": comment,
                    "timestamp": time.time()
                })
                # 设置过期时间
                self.redis_client.expire(comment_key, APP_CONFIG["redis"]["ttl"])
                
            # 3. 语义相似度去重（如果启用）
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
                    # 如果语义相似度计算失败，继续使用其他去重方法
                    
            # 通过所有去重检查，添加到结果列表
            unique_comments.append(comment)
            self.processed_comments.add(comment)
            
            # 限制内存中的评论数量
            if len(self.processed_comments) > self.max_local_cache:
                # 丢弃最旧的条目（转换为列表后移除第一个）
                self.processed_comments = set(list(self.processed_comments)[1:])
                
        return unique_comments
