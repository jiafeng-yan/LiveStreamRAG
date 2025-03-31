class PromptTemplates:
    """提示词模板"""

    @staticmethod
    def rag_prompt(question: str, context: str) -> str:
        """
        RAG提示词模板

        Args:
            question: 用户问题
            context: 检索到的上下文

        Returns:
            格式化的提示词
        """
        return f"""
我需要你回答有关某产品的问题。请仅使用以下提供的参考信息来回答问题。如果参考信息中没有相关内容，请明确说明你不知道答案，不要编造信息。

问题: {question}

参考信息:
{context}

请提供清晰、准确、有帮助的回答。
"""

    @staticmethod
    def system_prompt() -> str:
        """系统提示词"""
        return """
你是一个直播间的产品专家助手，负责根据产品知识库回答用户评论中的问题。
请保持回答简洁、专业、友好。
如果无法从知识库中找到答案，请诚实地说明，并引导用户咨询其他渠道。
不要编造不确定的信息。
"""
