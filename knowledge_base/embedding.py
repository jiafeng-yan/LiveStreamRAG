from sentence_transformers import SentenceTransformer
from typing import List, Any, Union
import numpy as np
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings


class EmbeddingModel:
    """文本嵌入模型封装类"""

    def __init__(self, model_name="all-MiniLM-L6-v2", device: str = "auto"):
        """
        初始化嵌入模型

        Args:
            model_name: 使用的模型名称，默认使用通用多语言模型
            device: 设备，默认使用自动选择
        """
        self.model = SentenceTransformer(model_name)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        try:
            self.embedding_function: Embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"嵌入模型 {self.model_name} 初始化成功在 {self.device}")
        except Exception as e:
            print(f"初始化嵌入模型 {self.model_name} 失败: {e}")
            raise

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

    def get_embedding_function(self) -> Embeddings:
        """返回内部持有的LangChain embedding对象"""
        if not hasattr(self, 'embedding_function'):
            raise AttributeError("Embedding function not initialized properly.")
        return self.embedding_function


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

# from langchain.embeddings import SentenceTransformerEmbeddings
# class ChromaEmbeddingAdapter(SentenceTransformerEmbeddings):
#     """适配Chroma数据库的嵌入适配器"""
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _embed_documents(self, texts):
#         return super().embed_documents(texts)  

#     def __call__(self, input):
#         return self._embed_documents(input)  
    

    # def __init__(self, model_name="all-MiniLM-L6-v2"):
    #     """初始化适配器"""
    #     self.model = SentenceTransformer(model_name)
    #     self.model_name = model_name

    # def __call__(self, input: List[str]) -> List[List[float]]:
    #     """
    #     实现Chroma所需的嵌入函数接口

    #     Args:
    #         input: 需要嵌入的文本列表

    #     Returns:
    #         嵌入向量列表
    #     """
    #     embeddings = self.model.encode(input, show_progress_bar=True)
    #     return embeddings.tolist()

    # def embed_documents(self, texts: List[str]) -> List[List[float]]:
    #     """
    #     为满足LangChain接口要求添加的方法

    #     Args:
    #         texts: 需要嵌入的文本列表

    #     Returns:
    #         嵌入向量列表
    #     """
    #     return self(texts)

    # def embed_query(self, text: str) -> List[float]:
    #     """
    #     为满足LangChain接口要求添加的方法

    #     Args:
    #         text: 查询文本

    #     Returns:
    #         嵌入向量
    #     """
    #     return self([text])[0]
