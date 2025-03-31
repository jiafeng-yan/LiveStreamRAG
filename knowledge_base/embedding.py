from sentence_transformers import SentenceTransformer
from typing import List, Any, Union
import numpy as np


class EmbeddingModel:
    """文本嵌入模型封装类"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        初始化嵌入模型

        Args:
            model_name: 使用的模型名称，默认使用通用多语言模型
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文本转换为嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为嵌入向量

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        return self.model.encode(text).tolist()


class ChromaEmbeddingAdapter:
    """适配Chroma数据库的嵌入适配器"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """初始化适配器"""
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        实现Chroma所需的嵌入函数接口

        Args:
            input: 需要嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        embeddings = self.model.encode(input, show_progress_bar=True)
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为满足LangChain接口要求添加的方法

        Args:
            texts: 需要嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        为满足LangChain接口要求添加的方法

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        return self([text])[0]
