import cv2
import time
import numpy as np
from paddleocr import PaddleOCR
from multiprocessing import Pool, Queue
import redis
import os

# ----------------- 配置参数 -------------------
CAPTURE_REGION = (360, 1200, 500, 450)  # 评论区坐标(y1,x1,h,w)
OCR_WORKERS = 4                        # 并行OCR进程数
REDIS_HOST = 'localhost'               # 去重数据库
FRAME_INTERVAL = 0.3                   # 截帧间隔(秒)
VIDEO_PATH = "data/video/short.mp4"     # 测试视频文件路径
USE_VIDEO = True                       # 是否使用视频文件测试
skip_frames = 10                         # 设置跳过的帧数
DEBUG_MODE = True                      # 调试模式
OCR_CONFIDENCE = 0.5                   # OCR置信度阈值（降低以获取更多结果）
# REDIS_HOME = 'E:/Redis/Redis-x64-5.0.14.1/redis-server.exe'
# ---------------------------------------------

class LiveCommentHunter:
    def __init__(self):
        self.ocr = PaddleOCR(use_gpu=True, lang='ch')  # 使用中文模型
        self.redis = redis.StrictRedis(REDIS_HOST)
        self.pool = Pool(OCR_WORKERS)
        self.frame_queue = Queue(maxsize=10)
        self.processed_frames = 0
        self.detected_texts = 0

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
                debug_dir = "data/debug"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/roi_{self.processed_frames}.jpg", roi)
            
            # 图像增强处理
            # 1. 转换为灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 2. 自适应阈值处理
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # 3. 降噪
            denoised = cv2.fastNlMeansDenoising(binary)
            
            if DEBUG_MODE:
                cv2.imwrite(f"{debug_dir}/enhanced_{self.processed_frames}.jpg", denoised)
                
            return denoised
        except Exception as e:
            print(f"图像处理错误: {str(e)}")
            return None

    def ocr_worker(self, img):
        """GPU加速的OCR处理"""
        try:
            if DEBUG_MODE:
                print(f"正在处理第 {self.processed_frames} 帧...")
            
            results = self.ocr.ocr(img, cls=False)
            # print(results)

            # results = [[
            #     [[[114.0, 5.0], [338.0, 7.0], [337.0, 37.0], [113.0, 35.0]], ('云山永春：阿蔡这么严肃', 0.9597381353378296)], 
            #     [[[115.0, 58.0], [286.0, 58.0], [286.0, 81.0], [115.0, 81.0]], ('白衣枪手想吃沙县了', 0.9847292304039001)], 
            #     [[[115.0, 103.0], [359.0, 103.0], [359.0, 126.0], [115.0, 126.0]], ('恐怖喵ovo贡献了几百个赞，', 0.9492839574813843)], 
            #     [[[111.0, 135.0], [155.0, 135.0], [155.0, 162.0], [111.0, 162.0]], ('丈人', 0.7722557783126831)], 
            #     [[[120.0, 185.0], [364.0, 185.0], [364.0, 208.0], [120.0, 208.0]], ('D11Qwerty 小橙子09！', 0.7425412535667419)], 
            #     [[[118.0, 230.0], [393.0, 230.0], [393.0, 253.0], [118.0, 253.0]], ('33蔡大佬六分仪犹大：茄', 0.9504022598266602)], 
            #     [[[115.0, 260.0], [227.0, 260.0], [227.0, 284.0], [115.0, 284.0]], ('方法做，无敌', 0.9936427474021912)], 
            #     [[[128.0, 312.0], [222.0, 316.0], [221.0, 336.0], [127.0, 333.0]], ('北欧州', 0.8537957072257996)], 
            #     [[[118.0, 352.0], [393.0, 353.0], [393.0, 376.0], [118.0, 375.0]], ('9】胸毛糙汉菜芯家里做怎么好', 0.8991268873214722)], 
            #     [[[115.0, 405.0], [322.0, 405.0], [322.0, 429.0], [115.0, 429.0]], ('职业法师老陈进入直播间', 0.9926782846450806)]
            # ]]
            
            if results is None or len(results) == 0:
                if DEBUG_MODE:
                    print("未检测到文本")
                return []
            
            detected_texts = []
            for line in results[0]:
                text, confidence = line[1]
                if confidence > OCR_CONFIDENCE:
                    detected_texts.append(text)
                    if DEBUG_MODE:
                        print(f"检测到文本: {text} (置信度: {confidence:.2f})")
            
            return detected_texts
        except Exception as e:
            print(f"OCR处理错误: {str(e)}")
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
                
            return change_ratio > 0.1  # 降低阈值到10%
        except Exception as e:
            print(f"运动检测错误: {str(e)}")
            return True

    def process_frame(self):
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

                # 3. 并行OCR识别
                self.processed_frames += 1
                results = self.ocr_worker(processed)
                self.handle_results(results)

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
            self.pool.close()
            self.pool.join()
            
            if DEBUG_MODE:
                print("\n处理统计：")
                print(f"总处理帧数: {self.processed_frames}")
                print(f"检测到文本数: {self.detected_texts}")

    def handle_results(self, comments):
        """处理识别结果"""
        for comment in comments:
            if not self.deduplicate(comment):
                print(f"新评论: {comment}")
                self.detected_texts += 1

    def deduplicate(self, comment):
        """基于布隆过滤器的去重"""
        bloom_key = f"comment_bloom:{hash(comment) % 1000000007}"
        if not self.redis.getbit(bloom_key, 1):
            self.redis.setbit(bloom_key, 1, 1)
            return False
        return True

if __name__ == "__main__":
    hunter = LiveCommentHunter()
    hunter.process_frame()