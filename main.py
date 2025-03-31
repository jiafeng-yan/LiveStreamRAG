import asyncio
import os
import time
from dotenv import load_dotenv
from config.app_config import APP_CONFIG
from screen_capture.capture import ScreenCapture
from screen_capture.ocr import OCRProcessor
from knowledge_base.embedding import EmbeddingModel
from knowledge_base.vector_store import VectorStore
from knowledge_base.document_loader import DocumentLoader
from llm_interface.llm_client import OpenRouterClient
from rag_engine.retrieval import Retriever
from rag_engine.generation import RAGEngine
from chat_response.response_formatter import ResponseFormatter
from chat_response.output_handler import OutputHandler

# 加载环境变量
load_dotenv()


async def process_comment(comment, rag_engine, output_handler):
    """处理单条评论"""
    try:
        # 生成回复
        response = await rag_engine.generate_response(comment)

        # 格式化回复
        formatted_response = ResponseFormatter.format_response(response, comment)
        truncated_response = ResponseFormatter.truncate_if_needed(formatted_response)

        # 输出和记录
        output_handler.print_response(comment, truncated_response)
        output_handler.log_interaction(comment, truncated_response)

    except Exception as e:
        print(f"处理评论时出错: {e}")


async def load_knowledge_base(vector_store, data_dir="data/documents"):
    """加载知识库"""
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
        vector_store.add_documents(documents, "initial_docs")
        print(f"已将 {len(documents)} 个文档添加到知识库")
    else:
        print("没有找到文档，请将产品文档放入 data/documents 目录")


async def main():
    """主函数"""
    print("初始化直播评论RAG系统...")

    # 创建输出处理器
    output_handler = OutputHandler()

    # 初始化知识库
    print("初始化知识库...")
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(
        embedding_model,
        APP_CONFIG["knowledge_base"]["db_path"],
        APP_CONFIG["knowledge_base"]["chunk_size"],
        APP_CONFIG["knowledge_base"]["chunk_overlap"],
    )

    # 加载知识库
    await load_knowledge_base(vector_store)

    # 初始化LLM客户端
    llm_client = OpenRouterClient(
        api_key=APP_CONFIG["llm"]["api_key"], model=APP_CONFIG["llm"]["model"]
    )

    # 初始化检索器
    retriever = Retriever(
        vector_store,
        APP_CONFIG["rag"]["top_k"],
        APP_CONFIG["rag"]["similarity_threshold"],
    )

    # 初始化RAG引擎
    rag_engine = RAGEngine(retriever, llm_client)

    # 初始化屏幕捕获器
    screen_capture = ScreenCapture(APP_CONFIG["capture"]["region"])

    # 初始化OCR处理器
    ocr_processor = OCRProcessor(APP_CONFIG["capture"]["ocr_lang"])

    print("系统初始化完成，开始监控直播评论...")

    # 主循环
    try:
        while True:
            # 捕获屏幕
            screenshot_path = screen_capture.capture_and_save()

            # 提取评论
            comments = ocr_processor.process_image(screenshot_path)

            # 处理新评论
            if comments:
                print(f"检测到 {len(comments)} 条新评论")
                for comment in comments:
                    asyncio.create_task(
                        process_comment(comment, rag_engine, output_handler)
                    )

            # 删除截图文件(可选)
            os.remove(screenshot_path)

            # 等待下一次捕获
            await asyncio.sleep(APP_CONFIG["capture"]["interval"])

    except KeyboardInterrupt:
        print("程序已停止")


if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("data/screenshots", exist_ok=True)
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    print("启动RAG直播评论系统...")
    print("请将产品文档放入 data/documents 目录")
    print("按Ctrl+C停止程序")

    asyncio.run(main())
