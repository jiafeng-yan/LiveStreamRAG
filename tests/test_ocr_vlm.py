import cv2
import time
import numpy as np
import os
import json
import requests
from PIL import Image
import base64
from io import BytesIO

# ----------------- 配置参数 -------------------
CAPTURE_REGION = (360, 1200, 500, 450)  # 评论区坐标(y1,x1,h,w)
VIDEO_PATH = "data/video/short.mp4"     # 测试视频文件路径
USE_VIDEO = True                       # 是否使用视频文件测试
skip_frames = 20                       # 设置跳过的帧数
DEBUG_MODE = True                      # 调试模式
FRAME_INTERVAL = 0.3                   # 截帧间隔(秒)
# OpenRouter配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")    # 替换为你的API密钥

# 可用的VLM模型列表
VLM_MODELS = [
    "qwen/qwen2.5-vl-32b-instruct:free",
    "gpt4vision/gpt-4-vision-preview:free",
    "google/gemini-pro-vision:free",
    "mistralai/mistral-medium:free"
]
# ---------------------------------------------

class VLMCommentHunter:
    def __init__(self):
        self.processed_frames = 0
        self.detected_texts = 0
        self.current_model_index = 0
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/yourproject",
        }
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.retry_count = 0
        self.max_retries = 3

    def get_next_model(self):
        """获取下一个可用的模型"""
        self.current_model_index = (self.current_model_index + 1) % len(VLM_MODELS)
        return VLM_MODELS[self.current_model_index]

    def smart_capture(self, frame):
        """智能区域截取+图像增强"""
        try:
            # 确保坐标不越界
            height, width = frame.shape[:2]
            y1 = min(max(CAPTURE_REGION[0], 0), height)
            x1 = min(max(CAPTURE_REGION[1], 0), width)
            h = min(CAPTURE_REGION[2], height - y1)
            w = min(CAPTURE_REGION[3], width - x1)
            
            roi = frame[y1:y1+h, x1:x1+w]
            
            if DEBUG_MODE:
                # 保存ROI区域图像用于调试
                debug_dir = "data/debug_vlm"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/roi_{self.processed_frames}.jpg", roi)
            
            # 图像增强处理
            # 1. 对比度增强
            enhanced = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)
            # 2. 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            if DEBUG_MODE:
                cv2.imwrite(f"{debug_dir}/enhanced_{self.processed_frames}.jpg", enhanced)
                
            return enhanced
        except Exception as e:
            print(f"图像处理错误: {str(e)}")
            return None

    def image_to_base64(self, image):
        """将OpenCV图像转换为base64字符串"""
        try:
            # 将OpenCV的BGR格式转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转换为PIL Image
            pil_image = Image.fromarray(image_rgb)
            # 保存为JPEG格式的字节流
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            # 转换为base64字符串
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            print(f"图像转换错误: {str(e)}")
            return None

    async def process_image_with_vlm(self, image):
        """使用VLM处理图像并提取文本"""
        while self.retry_count < self.max_retries:
            try:
                # 转换图像为base64
                base64_image = self.image_to_base64(image)
                if not base64_image:
                    return []

                current_model = VLM_MODELS[self.current_model_index]
                
                # 构建请求
                payload = {
                    "model": current_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "请识别图片中的所有中文评论文本。只需要返回评论文本内容，每条评论占一行，不需要其他解释。如果没有检测到评论文本，请返回空列表 []。"
                                },
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_image}"
                                }
                            ]
                        }
                    ]
                }

                # 发送请求
                response = requests.post(self.api_url, json=payload, headers=self.headers)
                
                if response.status_code == 429:  # 速率限制
                    print(f"模型 {current_model} 达到使用限制，尝试切换到下一个模型...")
                    self.retry_count += 1
                    next_model = self.get_next_model()
                    print(f"切换到模型: {next_model}")
                    continue
                
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                if DEBUG_MODE:
                    print(f"使用模型 {current_model} 的响应:", result)
                
                if 'choices' in result and len(result['choices']) > 0:
                    text = result['choices'][0]['message']['content']
                    # 将文本按行分割并过滤空行
                    comments = [line.strip() for line in text.split('\n') if line.strip()]
                    if DEBUG_MODE:
                        print(f"检测到 {len(comments)} 条评论:")
                        for comment in comments:
                            print(f"  - {comment}")
                    self.retry_count = 0  # 重置重试计数
                    return comments
                return []

            except Exception as e:
                print(f"VLM处理错误: {str(e)}")
                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    next_model = self.get_next_model()
                    print(f"发生错误，切换到模型: {next_model}")
                    continue
                break
        
        self.retry_count = 0  # 重置重试计数
        return []

    def motion_detect(self, prev, current):
        """帧间运动检测算法"""
        try:
            if prev is None or current is None:
                return True
                
            # 转换为灰度图
            if len(prev.shape) == 3:
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            if len(current.shape) == 3:
                current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                
            diff = cv2.absdiff(prev, current)
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            change_ratio = np.sum(diff) / diff.size
            
            if DEBUG_MODE:
                print(f"变化率: {change_ratio:.3f}")
                
            return change_ratio > 10  # 降低阈值到10%
        except Exception as e:
            print(f"运动检测错误: {str(e)}")
            return True

    async def process_frame(self):
        """核心处理流程"""
        prev_frame = None
        frame_count = 0
        
        if USE_VIDEO:
            if not os.path.exists(VIDEO_PATH):
                print(f"错误：视频文件 {VIDEO_PATH} 不存在")
                return
            cap = cv2.VideoCapture(VIDEO_PATH)
            print(f"已加载测试视频：{VIDEO_PATH}")
            print(f"视频信息：")
            print(f"- 总帧数：{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            print(f"- FPS：{cap.get(cv2.CAP_PROP_FPS):.2f}")
            print(f"- 分辨率：{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        else:
            cap = cv2.VideoCapture(0)
            print("已启动摄像头捕获")

        try:
            last_process_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    if USE_VIDEO:
                        print("视频播放完毕")
                        print(f"总计处理帧数: {self.processed_frames}")
                        print(f"检测到文本数: {self.detected_texts}")
                        break
                    continue

                frame_count += 1
                if frame_count % skip_frames != 0:  # 跳过指定数量的帧
                    continue

                # 控制处理帧率
                current_time = time.time()
                if current_time - last_process_time < FRAME_INTERVAL:
                    continue
                last_process_time = current_time

                # 1. 智能区域截取
                processed = self.smart_capture(frame)
                if processed is None:
                    continue
                
                # 2. 运动检测减少重复处理
                if prev_frame is not None and not self.motion_detect(prev_frame, processed):
                    if DEBUG_MODE:
                        print("跳过静态帧")
                    continue
                prev_frame = processed.copy()

                # 3. VLM处理
                self.processed_frames += 1
                comments = await self.process_image_with_vlm(processed)
                self.detected_texts += len(comments)

                # 显示处理区域（调试用）
                cv2.rectangle(frame, 
                            (CAPTURE_REGION[1], CAPTURE_REGION[0]),
                            (CAPTURE_REGION[1] + CAPTURE_REGION[3], 
                             CAPTURE_REGION[0] + CAPTURE_REGION[2]),
                            (0, 255, 0), 2)
                
                # 显示帧计数
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Processing Region', frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if DEBUG_MODE:
                print("\n处理统计：")
                print(f"总处理帧数: {self.processed_frames}")
                print(f"检测到文本数: {self.detected_texts}")

if __name__ == "__main__":
    import asyncio
    hunter = VLMCommentHunter()
    asyncio.run(hunter.process_frame()) 