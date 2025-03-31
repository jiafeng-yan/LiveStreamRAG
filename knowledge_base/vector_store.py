from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List, Dict, Any, Optional
import os
import shutil

from knowledge_base.embedding import ChromaEmbeddingAdapter


class VectorStore:
    """向量存储管理类"""

    def __init__(
        self,
        embedding_model,
        db_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        初始化向量存储

        Args:
            embedding_model: 嵌入模型
            db_path: 向量数据库存储路径
            chunk_size: 文本分割大小
            chunk_overlap: 分割重叠大小
        """
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 创建存储目录
        os.makedirs(db_path, exist_ok=True)

        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # 创建与Chroma兼容的嵌入适配器
        self.embeddings = ChromaEmbeddingAdapter("all-MiniLM-L6-v2")

        # 初始化向量存储
        self.vectordb = Chroma(
            persist_directory=db_path, embedding_function=self.embeddings
        )

        # 文档索引，用于管理文档和向量存储的关系
        self.document_index = {}

    def add_documents(
        self, documents: List[Document], doc_id: Optional[str] = None
    ) -> str:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表
            doc_id: 文档ID，如果不提供则自动生成

        Returns:
            文档ID
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.document_index) + 1}"

        # 分割文档
        chunks = self.text_splitter.split_documents(documents)

        # 为每个分块添加文档ID
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id

        # 添加到向量存储
        ids = self.vectordb.add_documents(chunks)

        # 更新文档索引
        self.document_index[doc_id] = {
            "chunk_ids": ids,
            "source": documents[0].metadata.get("source", "未知来源"),
            "chunks": len(chunks),
        }

        # 持久化存储
        self.vectordb.persist()

        return doc_id

    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档

        Args:
            doc_id: 文档ID

        Returns:
            是否成功删除
        """
        if doc_id not in self.document_index:
            return False

        # 获取文档对应的chunk IDs
        chunk_ids = self.document_index[doc_id]["chunk_ids"]

        # 从向量存储中删除
        self.vectordb.delete(chunk_ids)

        # 更新文档索引
        del self.document_index[doc_id]

        # 持久化存储
        self.vectordb.persist()

        return True

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似文档列表
        """
        return self.vectordb.similarity_search(query, k=k)

    def reset(self) -> None:
        """重置向量存储"""
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        os.makedirs(self.db_path, exist_ok=True)

        self.vectordb = Chroma(
            persist_directory=self.db_path, embedding_function=self.embeddings
        )
        self.document_index = {}
