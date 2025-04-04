from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List, Dict, Any, Optional
import os
import shutil
import traceback
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from knowledge_base.embedding import EmbeddingModel
from langchain.embeddings.base import Embeddings


class VectorStore:
    """向量存储管理类"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        db_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        初始化向量存储

        Args:
            embedding_model: 嵌入模型
            db_path: 向量数据库存储路径
            chunk_size: 文本分割大小
            chunk_overlap: 分割重叠大小
        """
        self.embedding_model = embedding_model
        self.embeddings: Embeddings = self.embedding_model.get_embedding_function()
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = embedding_model_name

        # 创建存储目录
        os.makedirs(db_path, exist_ok=True)

        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "-", "\n", "。", "？", "！", "(?<=\. )", " ", ""]
        )

        # 2. 定义或获取 ChromaDB 理解的嵌入函数
        # 使用 chromadb 内置的 SentenceTransformer 嵌入函数辅助类
        # 这确保了与 ChromaDB 底层的兼容性
        print(f"为 ChromaDB 内部配置嵌入函数元数据，模型: {self.model_name}")
        self.chroma_embedding_function_for_creation = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )

        # 初始化向量存储
        self.vectordb = self._load_or_create_db()

        # 文档索引，用于管理文档和向量存储的关系
        self.document_index = {}

    def _load_or_create_db(self):
        # 根据模型名称创建子目录路径
        model_dir_name = os.path.basename(str(self.model_name)).split('/')[-1]
        persist_path = os.path.join(self.db_path, model_dir_name)
        collection_name = "chroma_docs_collection"
        print(f"向量数据库路径: {persist_path}")
        print(f"使用的集合名称: {collection_name}")

        try:
            # 1. 使用 chromadb.PersistentClient 获取客户端
            print(f"尝试连接或创建持久化客户端于: {persist_path}")
            chroma_client = chromadb.PersistentClient(path=persist_path)

            # 3. 获取或创建集合
            # 使用 client.get_or_create_collection，如果不存在会自动创建
            # 需要传递集合名称和嵌入函数
            print(f"尝试获取或创建集合: {collection_name}")
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                # embedding_function=self.chroma_embedding_function_for_creation,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"成功获取或创建集合: {collection.name} (ID: {collection.id})")
            print(f"集合中的文档数量: {collection.count()}")

            # 4. 初始化 LangChain 的 Chroma 类
            # **将 LangChain 的嵌入函数传递给 LangChain Chroma 对象**
            print("初始化 LangChain Chroma 包装器 (传入 LangChain 的 embedding_function)...")
            vector_db = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )
            print("LangChain Chroma 包装器初始化成功。")
            return vector_db

        except Exception as e:
            print(f"加载或创建 ChromaDB 时出错: {e}")
            traceback.print_exc()
            raise RuntimeError("无法初始化向量数据库")

    def add_documents(
        self, documents: List[Document], source_id: Optional[str] = None
    ) -> str:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表
            source_id: 文档来源标识（用于生成块ID），如果不提供则使用默认值

        Returns:
            实际使用的文档来源标识
        """
        if not documents:
            print("没有文档需要添加。")
            return source_id or "default_source" # 返回一个标识符

        if source_id is None:
            # 尝试从第一个文档的元数据获取 source，否则用默认值
            first_doc_source = documents[0].metadata.get("source", "unknown_source")
            source_id = f"src_{os.path.basename(first_doc_source)}_{len(self.document_index)}"

        print(f"正在为来源 '{source_id}' 添加 {len(documents)} 个文档...")

        # 分割文档
        chunks = self.text_splitter.split_documents(documents)
        print(f"文档被分割成 {len(chunks)} 个块。")
        # print(f'chunks: \n {[chunk + '\n' for chunk in chunks]}')
        print(f'chunks: \n {chunks}')

        # 为每个分块添加文档ID元数据，并生成唯一的块ID
        chunk_contents = []
        chunk_metadatas = []
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}_chunk_{i}"
            chunk.metadata["doc_id"] = source_id # 保留文档来源标识
            chunk.metadata["chunk_index"] = i   # 可选：添加块索引
            chunk_contents.append(chunk.page_content)
            chunk_metadatas.append(chunk.metadata)
            chunk_ids.append(chunk_id)


        if not chunk_ids:
            print("没有生成任何文本块，无法添加。")
            return source_id

        try:
            # 添加到向量存储，**不再传递 embedding_function 参数**
            # 因为 self.vectordb 初始化时已经配置了
            print(f"准备添加 {len(chunk_contents)} 个文本块 (IDs: {chunk_ids[0]}...)...")
            added_ids = self.vectordb.add_texts(
                texts=chunk_contents, metadatas=chunk_metadatas, ids=chunk_ids
            )

            # 更新文档索引（如果需要跟踪块）
            self.document_index[source_id] = {
                "chunk_ids": added_ids if added_ids else chunk_ids, # Chroma 可能返回添加的ID列表
                "source": documents[0].metadata.get("source", "未知来源"),
                "chunks": len(chunks),
            }
            print(f"成功添加 {len(added_ids if added_ids else chunk_ids)} 个文本块。")

            return source_id

        except Exception as e:
            print(f"添加文档时出错: {e}")
            traceback.print_exc()
            # 可以考虑部分成功的情况，或者直接抛出异常
            raise # 重新抛出异常，让调用者知道失败了

    def delete_document(self, doc_id: str) -> bool:
        """
        删除与指定文档ID关联的所有块

        Args:
            doc_id: 文档ID (之前在 add_documents 中使用的 source_id)

        Returns:
            是否成功删除（或文档不存在）
        """
        if doc_id not in self.document_index:
            print(f"文档 ID '{doc_id}' 不在索引中，无法删除。")
            return False

        # 获取文档对应的chunk IDs
        chunk_ids_to_delete = self.document_index[doc_id].get("chunk_ids")

        if not chunk_ids_to_delete:
            print(f"警告: 文档 ID '{doc_id}' 在索引中，但没有关联的 chunk IDs。")
            # 仍然尝试从索引中移除
            del self.document_index[doc_id]
            return True # 认为操作是"成功的"，因为它不再存在于索引中

        try:
            print(f"准备删除与文档 ID '{doc_id}' 关联的 {len(chunk_ids_to_delete)} 个块...")
            # 从向量存储中删除，使用 LangChain Chroma 对象的 delete 方法
            self.vectordb.delete(ids=chunk_ids_to_delete)
            print(f"已请求删除块 (IDs: {chunk_ids_to_delete[0]}...)。")

            # 更新文档索引
            del self.document_index[doc_id]
            print(f"已从内部索引中移除文档 ID '{doc_id}'。")

            return True

        except Exception as e:
            print(f"删除文档 '{doc_id}' 的块时出错: {e}")
            traceback.print_exc()
            return False

    def similarity_search(self, query: str, k: int = 3, score_threshold: float = 0.7) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 分数阈值

        Returns:
            相似文档列表
        """
        print(f"在向量数据库中搜索与 '{query[:50]}...' 相关的文档 (k={k}, threshold={score_threshold})...")
        try:
            # 使用 LangChain Chroma 对象的 similarity_search_with_relevance_scores
            # 它会使用内部推断出的嵌入函数来嵌入查询
            results_with_scores = self.vectordb.similarity_search_with_relevance_scores(
                query, k=k
            )
            print(f"原始检索结果（带分数）: \n{[(doc.page_content[:30]+'...', score) for doc, score in results_with_scores]}")

            # 过滤低于阈值的结阈
            filtered_results = [
                (doc, score) for doc, score in results_with_scores if score >= score_threshold
            ]

            if not filtered_results:
                print("没有找到足够相关的文档。")
                return []

            print(f"找到 {len(filtered_results)} 个相关文档 (分数 >= {score_threshold})。")
            # 返回文档对象列表
            return [doc for doc, score in filtered_results]

        except Exception as e:
            print(f"相似性搜索时出错: {e}")
            traceback.print_exc()
            return [] # 返回空列表表示失败

    def reset(self) -> None:
        """重置向量存储"""
        # 获取模型对应的持久化路径
        model_dir_name = os.path.basename(str(self.model_name)).split('/')[-1]
        persist_path = os.path.join(self.db_path, model_dir_name)
        print(f"正在重置数据库路径: {persist_path}")

        # 关闭现有的客户端连接（如果可能且安全的话，chromadb client 可能没有显式 close）
        # 尝试删除目录
        if os.path.exists(persist_path):
            try:
                # ChromaDB 可能有文件锁，先置空引用可能有助于释放
                self.vectordb = None
                # chroma_client 在 _load_or_create_db 作用域内，这里无法直接访问来关闭
                shutil.rmtree(persist_path)
                print(f"已删除数据库目录: {persist_path}")
            except Exception as e:
                print(f"删除数据库目录时出错: {e}。可能需要手动删除。")
                # 即使删除失败，也继续尝试重新创建
        else:
            print("数据库目录不存在，无需删除。")

        # 重新创建目录并初始化
        os.makedirs(persist_path, exist_ok=True)
        self.vectordb = self._load_or_create_db()
        self.document_index = {}
        print("向量存储已重置。")

    def get_retriever(self, k: int = 3, score_threshold: float = 0.7):
        """获取配置好的检索器"""
        # 确保 self.vectordb 已正确初始化
        if not self.vectordb:
            raise RuntimeError("Vector DB 未初始化，无法获取检索器。")
        return self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': k, 'score_threshold': score_threshold}
        )
