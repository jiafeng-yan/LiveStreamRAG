import pyautogui
import cv2
import numpy as np
import time
from PIL import Image
import os


class ScreenCapture:
    def __init__(self, region=None):
        """
        初始化屏幕捕获器

        Args:
            region: 截图区域 (x, y, width, height)，若为None则引导用户选择
        """
        self.region = region
        if region is None:
            self.region = self._select_region()

    def _select_region(self):
        """让用户选择屏幕区域"""
        print("请选择需要监控的直播评论区域:")
        print("1. 全屏截图将显示在新窗口")
        print("2. 请在图像中拖动鼠标选择评论区域")
        print("3. 按Enter确认选择，按ESC重新选择")

        # 屏幕截图
        screenshot = pyautogui.screenshot()
        screen = np.array(screenshot)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

        # 使用OpenCV窗口选择区域
        roi = cv2.selectROI("选择直播评论区域", screen, False)
        cv2.destroyAllWindows()

        print(f"已选择区域: {roi}")
        return roi

    def capture(self):
        """捕获指定区域的屏幕图像"""
        x, y, width, height = self.region
        screenshot = pyautogui.screenshot(region=self.region)
        return screenshot

    def save_image(self, image, filename):
        """保存图像到文件"""
        image.save(filename)

    def capture_and_save(self, directory="data/screenshots"):
        """捕获并保存图像"""
        os.makedirs(directory, exist_ok=True)
        timestamp = int(time.time())
        filename = f"{directory}/screenshot_{timestamp}.png"
        screenshot = self.capture()
        self.save_image(screenshot, filename)
        return filename
