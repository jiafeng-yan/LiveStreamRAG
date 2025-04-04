import pytesseract
from PIL import Image
import re
import os


class OCRProcessor:
    def __init__(self, lang="chi_sim+eng"):
        """
        初始化OCR处理器

        Args:
            lang: OCR语言设置
        """
        self.lang = lang
        self.previous_texts = set()  # 用于去重

    def extract_text(self, image_path):
        """从图像中提取文本"""
        try:
            text = pytesseract.image_to_string(Image.open(image_path), lang=self.lang, config="--psm 6")
            return text
        except Exception as e:
            print(f"OCR处理出错: {e}")
            return ""

    def extract_comments(self, text):
        """从文本中提取评论"""
        # 按行分割文本
        lines = text.split("\n")
        # 过滤空行
        lines = [line.strip() for line in lines if line.strip()]
        return lines

    def process_image(self, image_path):
        """处理图像并返回新的评论"""
        text = self.extract_text(image_path)
        comments = self.extract_comments(text)

        # 过滤已处理的评论
        new_comments = [
            comment for comment in comments if comment not in self.previous_texts
        ]

        # 更新已处理评论集合
        self.previous_texts.update(new_comments)

        # 保持集合大小在合理范围内
        if len(self.previous_texts) > 1000:
            self.previous_texts = set(list(self.previous_texts)[-500:])

        return new_comments
