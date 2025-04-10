import asyncio
import os
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "E:/data/huggingface"
import sys
from pathlib import Path
import traceback

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from dotenv import load_dotenv
from config.app_config import APP_CONFIG
from knowledge_base.embedding import EmbeddingModel
from knowledge_base.vector_store import VectorStore
from knowledge_base.document_loader import DocumentLoader
from llm_interface.llm_client import OpenRouterClient
from rag_engine.retrieval import Retriever
from rag_engine.generation import RAGEngine
from chat_response.response_formatter import ResponseFormatter
from chat_response.output_handler import OutputHandler


class RAGSystemTester:
    """RAG系统测试类"""

    def __init__(self):
        """初始化测试环境"""
        # 加载环境变量
        load_dotenv()

        # 检查API密钥
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("错误: 未设置OPENROUTER_API_KEY环境变量")
            print("请在.env文件中添加您的API密钥")
            print("示例: OPENROUTER_API_KEY=your_api_key_here")
            raise ValueError("缺少API密钥")
        else:
            print(f"已检测到API密钥: {api_key[:10]}...")

        # 创建输出处理器
        self.output_handler = OutputHandler()

        # 初始化系统组件
        self._initialize_components()

    def _initialize_components(self):
        """初始化系统组件"""
        print("初始化测试环境...")

        try:
            # 初始化知识库
            print("初始化知识库...")
            embedding_model = EmbeddingModel(APP_CONFIG["knowledge_base"]["embedding_model"])
            print(f"嵌入模型初始化成功")

            self.vector_store = VectorStore(
                embedding_model,
                APP_CONFIG["knowledge_base"]["db_path"],
                APP_CONFIG["knowledge_base"]["chunk_size"],
                APP_CONFIG["knowledge_base"]["chunk_overlap"],
                APP_CONFIG["knowledge_base"]["embedding_model"],
            )
            print("向量存储初始化成功")

            # 初始化LLM客户端
            print(f"正在初始化LLM客户端，模型: {APP_CONFIG['llm']['model']}")
            self.llm_client = OpenRouterClient(
                api_key=APP_CONFIG["llm"]["api_key"], model=APP_CONFIG["llm"]["model"]
            )
            print("LLM客户端初始化成功")

            # 初始化检索器
            self.retriever = Retriever(
                self.vector_store,
                APP_CONFIG["rag"]["top_k"],
                APP_CONFIG["rag"]["similarity_threshold"],
            )
            print("检索器初始化成功")

            # 初始化RAG引擎
            self.rag_engine = RAGEngine(self.retriever, self.llm_client)
            print("RAG引擎初始化成功")

        except Exception as e:
            print(f"初始化组件时出错: {str(e)}")
            traceback.print_exc()
            raise

        print("测试环境初始化完成")

    async def load_test_documents(self, data_dir="data/documents"):
        """加载测试文档到知识库"""
        os.makedirs(data_dir, exist_ok=True)

        # 检查是否有文档
        documents = []
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                try:
                    docs = DocumentLoader.load(file_path)
                    documents.extend(docs)
                    print(f"已加载文档: {filename}")
                except Exception as e:
                    print(f"加载文档 {filename} 时出错: {e}")

        if documents:
            self.vector_store.add_documents(documents, "test_docs")
            print(f"已将 {len(documents)} 个文档添加到测试知识库")
        else:
            print("警告: 没有找到文档，请将产品文档放入 data/documents 目录")

    async def process_comment(self, comment):
        """处理单条评论"""
        try:
            print(f"\n处理评论: {comment}")

            # 生成回复
            response = await self.rag_engine.generate_response(comment)

            # 格式化回复
            formatted_response = ResponseFormatter.format_response(response, comment)
            truncated_response = ResponseFormatter.truncate_if_needed(
                formatted_response
            )

            # 输出和记录
            self.output_handler.print_response(comment, truncated_response)
            self.output_handler.log_interaction(comment, truncated_response)

            return truncated_response

        except Exception as e:
            error_msg = f"处理评论时出错: {e}"
            print(error_msg)
            return error_msg

    async def test_comments(self, comments):
        """测试多条评论"""
        print(f"开始测试 {len(comments)} 条模拟评论...")

        results = []
        for comment in comments:
            response = await self.process_comment(comment)
            results.append((comment, response))
            # 稍微暂停以避免API速率限制
            await asyncio.sleep(1)

        return results

    async def interactive_test(self):
        """交互式测试模式"""
        print("\n=== 进入交互式测试模式 ===")
        print("输入评论进行测试，输入'exit'退出")

        while True:
            comment = input("\n请输入模拟评论: ")
            if comment.lower() == "exit":
                break

            await self.process_comment(comment)


async def main():
    """主函数"""
    # 确保测试目录存在
    data_root = 'data/documents'
    os.makedirs(data_root, exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    # 创建测试器
    tester = RAGSystemTester()

    # 加载测试文档
    await tester.load_test_documents(data_dir=data_root)

    # 预定义的测试评论
    test_comments = [
        "这个产品有什么特点？",
        # "哇咔咔老年奶粉适合什么年龄段的人喝？",
        # "这个奶粉有什么营养成分？",
        # "请问价格是多少？",
        # "这款奶粉和普通牛奶相比有什么优势？",
        # '这款奶粉适合糖尿病患者吗?',
        '这款奶粉的价格信息是？',
    ]

    # 测试预定义评论
    await tester.test_comments(test_comments)

    # 进入交互式测试模式
    await tester.interactive_test()


if __name__ == "__main__":
    asyncio.run(main())
