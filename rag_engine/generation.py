import asyncio
from llm_interface.prompt_templates import PromptTemplates
from rag_engine.retrieval import Retriever
from config import APP_CONFIG


class RAGEngine:
    """RAG引擎"""

    def __init__(self, retriever: Retriever, llm_client):
        """
        初始化RAG引擎

        Args:
            retriever: 检索器
            llm_client: LLM客户端
        """
        self.retriever = retriever
        self.llm_client = llm_client

    async def generate_response(self, query: str) -> str:
        """
        生成回复

        Args:
            query: 用户查询

        Returns:
            生成的回复
        """
        # 检索相关文档
        documents = self.retriever.retrieve(query)

        # 如果没有相关文档
        if not documents:
            # return await self.llm_client.generate(
            #     prompt=f"用户问题: {query}\n\n我没有找到与此问题相关的信息，请告知我无法回答。",
            #     system_prompt=PromptTemplates.system_prompt(),
            #     temperature=0.3,
            #     max_tokens=100,
            # )
            return "我没有找到与此问题相关的信息，请告知我无法回答。"

        # 格式化上下文
        context = self.retriever.format_context(documents)

        # 构建RAG提示词
        prompt = PromptTemplates.rag_prompt(query, context)

        # 调用LLM生成回复
        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=PromptTemplates.system_prompt(),
            temperature=0.3,
            max_tokens=500,
        )

        return response
