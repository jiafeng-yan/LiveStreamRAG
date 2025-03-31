import os
import time
from datetime import datetime


class OutputHandler:
    """输出处理器"""

    def __init__(self, log_dir: str = "data/logs"):
        """
        初始化输出处理器

        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志文件
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = f"{log_dir}/chat_log_{date_str}.txt"

    def print_response(self, query: str, response: str) -> None:
        """
        打印回复

        Args:
            query: 用户查询
            response: 生成的回复
        """
        print("\n" + "=" * 50)
        print(f"问题: {query}")
        print("-" * 50)
        print(f"回答: {response}")
        print("=" * 50 + "\n")

    def log_interaction(self, query: str, response: str) -> None:
        """
        记录交互到日志

        Args:
            query: 用户查询
            response: 生成的回复
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]\n")
            f.write(f"问题: {query}\n")
            f.write(f"回答: {response}\n")
            f.write("-" * 50 + "\n")
