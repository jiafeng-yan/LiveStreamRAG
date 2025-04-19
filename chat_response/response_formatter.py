import re
from config.app_config import APP_CONFIG


class ResponseFormatter:
    """
    响应格式化器，用于格式化LLM的输出
    """

    @staticmethod
    def format_response(response, query=None):
        """
        格式化回复，添加问题前缀
        
        Args:
            response: LLM生成的原始回复
            query: 用户的查询，用于添加前缀
        
        Returns:
            str: 格式化后的回复
        """
        if not response:
            return "抱歉，我无法生成回复。请稍后再试。"

        # 检查回复中是否已经包含"问题"和"回答"的标记
        if "问题:" in response or "回答：" in response:
            return response
            
        # 清理回复文本，去除多余空行
        cleaned_response = re.sub(r'\n{3,}', '\n\n', response.strip())
        
        # 如果有查询，添加问题前缀
        if query:
            formatted_response = f"问题: {query}\n\n回答: {cleaned_response}"
        else:
            formatted_response = cleaned_response
            
        return formatted_response

    @staticmethod
    def truncate_if_needed(response, max_length=None, min_keep_ratio=None):
        """
        如有必要，截断过长的回复
        
        Args:
            response: 格式化后的回复
            max_length: 最大长度，默认使用配置中的设置
            min_keep_ratio: 最小保留比例，默认使用配置中的设置
        
        Returns:
            str: 截断后的回复
        """
        if not max_length:
            max_length = APP_CONFIG["output"]["max_response_length"]
            
        if not min_keep_ratio:
            min_keep_ratio = APP_CONFIG["output"]["min_keep_ratio"]
            
        if len(response) <= max_length:
            return response
            
        # 计算保留的内容长度
        keep_length = int(max_length * min_keep_ratio)
        
        # 获取前后部分
        prefix_len = keep_length // 2
        suffix_len = keep_length - prefix_len
        
        # 截取前后部分
        prefix = response[:prefix_len]
        suffix = response[-suffix_len:]
        
        # 构建截断后的回复
        truncated = f"{prefix}\n...\n[内容过长，已截断]\n...\n{suffix}"
        
        return truncated
