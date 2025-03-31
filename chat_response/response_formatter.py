import re
from typing import Dict, Any


class ResponseFormatter:
    """回复格式化器"""

    @staticmethod
    def format_response(response: str, query: str) -> str:
        """
        格式化回复

        Args:
            response: 原始回复
            query: 用户查询

        Returns:
            格式化后的回复
        """
        # 去除多余空行
        response = re.sub(r"\n\s*\n", "\n\n", response)

        # 添加回复前缀
        formatted = f"问题：{query}\n\n回答：{response}"

        return formatted

    @staticmethod
    def truncate_if_needed(text: str, max_length: int = 1000) -> str:
        """
        如果回复过长，进行截断

        Args:
            text: 回复文本
            max_length: 最大长度

        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text

        # 尝试在句子边界截断
        truncated = text[:max_length]
        last_period = truncated.rfind("。")
        if last_period > max_length * 0.8:  # 确保至少保留80%的内容
            truncated = truncated[: last_period + 1]

        truncated += "\n...(回复较长，已截断)"
        return truncated
