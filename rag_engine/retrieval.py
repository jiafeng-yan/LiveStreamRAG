from typing import List, Dict, Any
from langchain.docstore.document import Document


class Retriever:
    """检索器"""

    def __init__(self, vector_store, top_k: int = 3, similarity_threshold: float = 0.5):
        """
        初始化检索器

        Args:
            vector_store: 向量存储
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本

        Returns:
            相关文档列表
        """
        return self.vector_store.similarity_search(query, k=self.top_k, score_threshold=self.similarity_threshold)

    def format_context(self, documents: List[Document]) -> str:
        """
        格式化检索结果为上下文字符串

        Args:
            documents: 检索到的文档列表

        Returns:
            格式化的上下文
        """
        context_parts = []

        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f"第{page}页" if page else ""

            context_parts.append(
                f"[参考文档 {i+1}] {source} {page_info}\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)
