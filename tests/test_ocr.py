import os
import sys
from pathlib import Path
import traceback

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from PIL import Image, ImageDraw, ImageFont
from screen_capture.ocr import OCRProcessor
from config.app_config import APP_CONFIG

def test_ocr_with_mock_images():
    # 创建测试目录
    test_dir = "data/test_screenshots"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建带有评论的模拟截图
    mock_comments = [
        "这个产品有什么特点？",
        "哇咔咔老年奶粉适合什么年龄段的人喝？",
        "这个奶粉有什么营养成分？",
        "请问价格是多少？",
        "这款奶粉和普通牛奶相比有什么优势？"
    ]
    
    # 创建模拟评论区图片
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # 尝试加载中文字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("simhei.ttf", 24)  # 尝试使用黑体
    except IOError:
        try:
            font = ImageFont.truetype("simsun.ttc", 24)  # 尝试使用宋体
        except IOError:
            font = ImageFont.load_default()  # 使用默认字体
    
    # 绘制评论
    y_position = 50
    for comment in mock_comments:
        d.text((50, y_position), comment, fill=(0, 0, 0), font=font)
        y_position += 50
    
    # 保存测试图片
    test_image_path = os.path.join(test_dir, "mock_comments.png")
    img.save(test_image_path)
    
    # 使用OCR处理器提取评论
    ocr_processor = OCRProcessor(lang=APP_CONFIG["capture"]['ocr_lang'])
    comments = ocr_processor.process_image(test_image_path)
    
    # 打印结果并计算准确率
    print("模拟评论:")
    for comment in mock_comments:
        print(f"  - {comment}")
    
    print("\nOCR 提取的评论:")
    for comment in comments:
        print(f"  - {comment}")
    
    # from difflib import SequenceMatcher
    # import re
    # def preprocess(text):
    #     """预处理文本：移除所有非汉字、字母、数字及下划线的字符"""
    #     return re.sub(r'[^\w\u4e00-\u9fff]', '', text)

    # a = preprocess(mock_comments)
    # b = preprocess(comments)
    # # match_rate = SequenceMatcher(None, mock_comments, comments).ratio()
    # match_rate = SequenceMatcher(None, a, b).ratio()
    # # 计算匹配率
    # # matches = sum(1 for c in comments if any(m in c or c in m for m in mock_comments))
    # # match_rate = matches / len(mock_comments) if mock_comments else 0
    # print(f"\n匹配率: {match_rate:.2%}")

if __name__ == "__main__":
    test_ocr_with_mock_images()