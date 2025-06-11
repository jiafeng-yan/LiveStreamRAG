import os
import sys
import time
import json
import base64
import asyncio
import aiohttp
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance
from config.app_config import APP_CONFIG
import hashlib

# 捕获区域设置 (x, y, width, height)
CAPTURE_REGION = (360, 1200, 500, 450)  # 评论区域坐标
VIDEO_PATH = "data/video/short.mp4"  # 视频文件路径
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# VLM模型配置
MODEL_CONFIGS = APP_CONFIG["ocr"]["vlm_models"]
VLM_MODELS = list(MODEL_CONFIGS.keys())

# 调试设置
DEBUG_MODE = True  # 启用调试模式，保存处理过的图像
DEBUG_DIR = "data/debug_vlm"  # 调试图像保存目录
os.makedirs(DEBUG_DIR, exist_ok=True)

class VLMCommentHunter:
    """使用VLM模型的评论识别器"""
    
    def __init__(self):
        """初始化评论识别器"""
        self.api_key = OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("缺少OPENROUTER_API_KEY环境变量")
            
        self.processed_comments = set()  # 已处理的评论
        self.current_model_index = 0
        self.max_retries = 3
        self.retry_count = 0
        self.api_url = APP_CONFIG["ocr"]["api_url"]
        self.motion_threshold = APP_CONFIG["ocr"]["performance_optimization"]["motion_threshold"] / 1000.0
        self.previous_frame = None
        
        # 检查是否需要跳帧
        self.skip_frames = APP_CONFIG["ocr"]["performance_optimization"]["skip_frames"]
        self.frame_counter = 0
            
    def get_current_model(self):
        """获取当前使用的VLM模型"""
        return VLM_MODELS[self.current_model_index]
        
    def get_next_model(self):
        """获取下一个VLM模型并更新索引"""
        self.current_model_index = (self.current_model_index + 1) % len(VLM_MODELS)
        current_model = self.get_current_model()
        print(f"切换到下一个模型: {current_model}")
        return current_model
        
    def get_model_payload(self, image_path, model_name):
        """根据当前模型生成请求负载"""
        model_config = MODEL_CONFIGS[model_name]
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
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": APP_CONFIG["ocr"]["temperature"]
        }
        
    async def process_image_with_vlm(self, image_path):
        """使用VLM处理图像提取评论"""
        retry_count = 0
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        while retry_count < self.max_retries:
            current_model = self.get_current_model()
            payload = self.get_model_payload(image_path, current_model)
            
            if DEBUG_MODE:
                print(f"使用模型 {current_model} 处理图像...")
                
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
                            self.get_next_model()
                            retry_count += 1
                            continue
                            
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"API请求失败: {response.status}, {error_text}")
                            self.get_next_model()
                            retry_count += 1
                            continue
                            
                        result = await response.json()
                        if DEBUG_MODE:
                            print(f"API响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
                            
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # 检查内容是否为空
                        if not content.strip():
                            print(f"模型 {current_model} 返回了空内容，切换到下一个模型...")
                            self.get_next_model()
                            retry_count += 1
                            continue
                            
                        # 成功获取内容，重置重试计数
                        retry_count = 0
                        return content
                        
            except Exception as e:
                print(f"处理图像时发生错误: {e}")
                self.get_next_model()
                retry_count += 1
                
        print(f"已达到最大重试次数 ({self.max_retries})，无法处理图像")
        return ""
        
    def preprocess_image(self, image_path):
        """图像预处理"""
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
            
            if DEBUG_MODE:
                debug_path = os.path.join(DEBUG_DIR, os.path.basename(image_path))
                img.save(debug_path)
                print(f"已保存预处理图像: {debug_path}")
                
        except Exception as e:
            print(f"图像预处理失败: {e}")
            
    def should_process_frame(self, frame):
        """判断是否应该处理当前帧"""
        # 跳帧检查
        self.frame_counter += 1
        if self.frame_counter % (self.skip_frames + 1) != 0:
            if DEBUG_MODE:
                print(f"跳过第 {self.frame_counter} 帧 (每 {self.skip_frames + 1} 帧处理一次)")
            return False
            
        # 运动检测
        if self.previous_frame is not None:
            diff = cv2.absdiff(frame, self.previous_frame)
            non_zero_count = np.count_nonzero(diff)
            avg_diff = non_zero_count / diff.size
            
            if avg_diff <= self.motion_threshold:
                if DEBUG_MODE:
                    print(f"跳过处理: 图像变化量({avg_diff:.5f})低于阈值({self.motion_threshold:.5f})")
                return False
        
        self.previous_frame = frame.copy()
        return True
        
    def clean_comments(self, comments_text):
        """清理并提取评论"""
        import re
        
        comments = []
        if not comments_text:
            return comments
            
        for line in comments_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # 跳过包含"问题"/"回答"的行
            if line.startswith("问题:") or line.startswith("回答:") or line.startswith("问题：") or line.startswith("回答："):
                continue
                
            # 去除数字开头，如 "1. ", "2) ", "[3]" 等
            line = re.sub(r"^\d+[\.\)\]]\s*", "", line)
            
            # 去除问题/回答前缀
            line = re.sub(r"^(问题|回答|Question|Answer)[：:]\s*", "", line)
            
            if line:
                comments.append(line)
                
        return comments
        
    def deduplicate_comments(self, comments):
        """去除重复评论"""
        unique_comments = []
        
        for comment in comments:
            # 跳过已处理的评论
            if comment in self.processed_comments:
                continue
                
            # 添加到结果并更新缓存
            unique_comments.append(comment)
            self.processed_comments.add(comment)
            
            # 限制缓存大小
            if len(self.processed_comments) > 1000:
                self.processed_comments = set(list(self.processed_comments)[100:])
                
        return unique_comments
        
    async def process_image(self, image_path):
        """处理图像并提取评论"""
        # 预处理图像
        self.preprocess_image(image_path)
        
        # 使用VLM提取评论
        comments_text = await self.process_image_with_vlm(image_path)
        
        # 清理并提取评论
        comments = self.clean_comments(comments_text)
        
        # 去重
        unique_comments = self.deduplicate_comments(comments)
        
        if DEBUG_MODE:
            print(f"提取到 {len(comments)} 条评论，去重后剩余 {len(unique_comments)} 条")
            
        return unique_comments

async def process_video(video_path : str = None):
    """从视频文件中提取评论"""
    if video_path is None:
        video_path = VIDEO_PATH
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    print(f"开始处理视频: {video_path}")
    
    # 初始化评论识别器
    comment_hunter = VLMCommentHunter()
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
        
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"视频帧率: {fps}fps, 总帧数: {frame_count}, 时长: {duration:.2f}秒")
    
    # 设置输出目录
    output_dir = "data/frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理视频帧
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 裁剪评论区域
        if CAPTURE_REGION:
            x, y, w, h = CAPTURE_REGION
            frame = frame[y:y+h, x:x+w]
            
        # 判断是否处理当前帧
        if not comment_hunter.should_process_frame(frame):
            frame_index += 1
            continue
            
        # 保存当前帧
        frame_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # 处理当前帧
        print(f"\n处理第 {frame_index} 帧...")
        comments = await comment_hunter.process_image(frame_path)
        
        # 打印提取的评论
        if comments:
            print("\n提取到的评论:")
            for i, comment in enumerate(comments, 1):
                print(f"{i}. {comment}")
        else:
            print("未提取到评论")
            
        # 删除临时文件
        if not DEBUG_MODE:
            os.remove(frame_path)
            
        frame_index += 1
        
    # 释放资源
    cap.release()
    print(f"视频处理完成，共处理 {frame_index} 帧")

async def video_mock(components, video_path : str = None):
    """从视频文件中提取评论"""
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    print(f"开始处理视频: {video_path}")
    
    # 初始化评论识别器
    # comment_hunter = VLMCommentHunter()

    rag_engine = components["rag_engine"]
    response_formatter = components["response_formatter"]
    output_handler = components["output_handler"]
    ocr_processor = components["ocr_processor"]
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
        
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"视频帧率: {fps}fps, 总帧数: {frame_count}, 时长: {duration:.2f}秒")
    
    # 设置输出目录
    output_dir = "data/frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理视频帧
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 裁剪评论区域
        if CAPTURE_REGION:
            x, y, w, h = CAPTURE_REGION
            frame = frame[y:y+h, x:x+w]
            
        # 判断是否处理当前帧
        # if not comment_hunter.should_process_frame(frame):
        frame_index += 1
        if frame_index % comment_hunter.skip_frames != 0:
            continue
            
        # 保存当前帧
        frame_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # 处理当前帧
        print(f"\n处理第 {frame_index} 帧...")
        # comments = await comment_hunter.process_image(frame_path)
        comments = await 
        
        # 打印提取的评论
        if comments:
            print("\n提取到的评论:")
            for i, comment in enumerate(comments, 1):
                print(f"{i}. {comment}")
        else:
            print("未提取到评论")
            
        # 删除临时文件
        if not DEBUG_MODE:
            os.remove(frame_path)
            
        frame_index += 1
        
    # 释放资源
    cap.release()
    print(f"视频处理完成，共处理 {frame_index} 帧")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("data/frames", exist_ok=True)
    os.makedirs("data/debug_vlm", exist_ok=True)
    
    print("开始VLM评论识别测试...")
    
    # 运行视频处理
    asyncio.run(process_video())