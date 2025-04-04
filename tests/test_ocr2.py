import cv2
import time
import numpy as np
from paddleocr import PaddleOCR
from multiprocessing import Pool, Queue
import redis

# ----------------- 配置参数 -------------------
CAPTURE_REGION = (360, 880, 600, 200)  # 评论区坐标(y1,x1,h,w)
OCR_WORKERS = 4                        # 并行OCR进程数
REDIS_HOST = 'localhost'               # 去重数据库
FRAME_INTERVAL = 0.3                   # 截帧间隔(秒)
# ---------------------------------------------

class LiveCommentHunter:
    def __init__(self):
        self.ocr = PaddleOCR(use_gpu=True)  # 使用PaddleOCR GPU加速
        self.redis = redis.StrictRedis(REDIS_HOST)
        self.pool = Pool(OCR_WORKERS)
        self.frame_queue = Queue(maxsize=10)

    def smart_capture(self, frame):
        """智能区域截取+图像增强"""
        # 使用边缘检测确保评论区定位准确
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        roi = frame[CAPTURE_REGION[0]:CAPTURE_REGION[0]+CAPTURE_REGION[2],
                    CAPTURE_REGION[1]:CAPTURE_REGION[1]+CAPTURE_REGION[3]]
        # 图像增强处理
        roi = cv2.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)
        return cv2.convertScaleAbs(roi, alpha=1.5, beta=40)

    def motion_detect(self, prev, current):
        """帧间运动检测算法"""
        diff = cv2.absdiff(prev, current)
        diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        return np.sum(diff) / diff.size > 0.2  # 变化超过20%认为有更新

    def ocr_worker(self, img):
        """GPU加速的OCR处理"""
        results = self.ocr.ocr(img, cls=True)
        return [line[1][0] for line in results if line[1][1] > 0.8]  # 置信度过滤

    def deduplicate(self, comment):
        """基于布隆过滤器的去重"""
        bloom_key = f"comment_bloom:{hash(comment) % 1000000007}"
        if not self.redis.getbit(bloom_key, 1):
            self.redis.setbit(bloom_key, 1, 1)
            return False
        return True

    def process_frame(self):
        """核心处理流程"""
        prev_frame = None
        cap = cv2.VideoCapture(0)  # 接入直播流

        while True:
            ret, frame = cap.read()
            if not ret: continue

            # 1. 智能区域截取
            processed = self.smart_capture(frame)
            
            # 2. 运动检测减少重复处理
            if prev_frame is not None and \
               not self.motion_detect(prev_frame, processed):
                continue
            prev_frame = processed.copy()

            # 3. 并行OCR识别
            self.pool.apply_async(self.ocr_worker, (processed,),
                                  callback=self.handle_results)

    def handle_results(self, comments):
        """处理识别结果"""
        for comment in comments:
            if not self.deduplicate(comment):
                print(f"New Comment: {comment}")
                # 存储到数据库或文件

if __name__ == "__main__":
    hunter = LiveCommentHunter()
    hunter.process_frame()