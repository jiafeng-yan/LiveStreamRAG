#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from pathlib import Path

# 确保当前目录在路径中，以便导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入相关模块
from knowledge_base.document_loader import DocumentLoader
from knowledge_base.vector_store import VectorStore
from llm_interface.llm_client import LLMClient
from rag_engine.retrieval import Retriever
from rag_engine.generation import RAGEngine
from chat_response.output_handler import OutputHandler
from chat_response.response_formatter import ResponseFormatter
from screen_capture.capture import ScreenCapture
from screen_capture.ocr import OCRProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("data/logs/app.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("main")

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        "data/logs",
        "data/documents",
        "data/vector_store",
        "data/debug_vlm",
        "data/video"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    logger.info("目录结构已创建")

def initialize_system():
    """初始化系统组件"""
    # 初始化知识库和向量存储
    document_loader = DocumentLoader()
    vector_store = VectorStore()
    
    # 初始化LLM客户端
    llm_client = LLMClient()
    
    # 初始化检索器和RAG引擎
    retriever = Retriever(vector_store)
    rag_engine = RAGEngine(retriever, llm_client)
    
    # 初始化响应处理组件
    response_formatter = ResponseFormatter()
    output_handler = OutputHandler()
    
    # 初始化屏幕捕获和OCR组件
    screen_capture = ScreenCapture()
    ocr_processor = OCRProcessor()
    
    logger.info("系统组件已初始化")
    
    return {
        "document_loader": document_loader,
        "vector_store": vector_store,
        "llm_client": llm_client,
        "retriever": retriever,
        "rag_engine": rag_engine,
        "response_formatter": response_formatter,
        "output_handler": output_handler,
        "screen_capture": screen_capture,
        "ocr_processor": ocr_processor
    }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM_RAG系统 - 基于大语言模型的检索增强生成系统")
    
    parser.add_argument("--load-documents", action="store_true", help="加载data/documents目录下的文档到知识库")
    parser.add_argument("--interactive", action="store_true", help="启动交互式问答模式")
    parser.add_argument("--capture-mode", action="store_true", help="启动屏幕捕获模式")
    parser.add_argument("--video-file", type=str, help="使用视频文件代替屏幕捕获")
    
    return parser.parse_args()

def interactive_mode(components):
    """交互式问答模式"""
    rag_engine = components["rag_engine"]
    response_formatter = components["response_formatter"]
    output_handler = components["output_handler"]
    
    print("\n===== 交互式问答模式 =====")
    print("输入 'exit' 或 'quit' 退出\n")
    
    while True:
        query = input("\n请输入问题: ")
        if query.lower() in ["exit", "quit", "退出"]:
            break
            
        # 生成回答
        response = rag_engine.generate_response(query)
        
        # 格式化回答
        formatted_response = response_formatter.format_response(response, query)
        
        # 输出回答
        output_handler.print_response(query, formatted_response)
        output_handler.log_interaction(query, formatted_response)

def capture_mode(components, video_file=None):
    """屏幕捕获模式"""
    from tests.test_ocr_vlm import run_vlm_test
    
    if video_file:
        print(f"\n===== 视频文件处理模式 =====")
        print(f"视频文件: {video_file}")
        run_vlm_test(video_path=video_file)
    else:
        print("\n===== 屏幕捕获模式 =====")
        run_vlm_test()

def load_documents(components):
    """加载文档到知识库"""
    document_loader = components["document_loader"]
    vector_store = components["vector_store"]
    
    print("\n===== 加载文档到知识库 =====")
    
    documents_dir = Path("data/documents")
    if not documents_dir.exists() or not any(documents_dir.iterdir()):
        print("未找到文档，请将文档放入 data/documents 目录")
        return
        
    # 加载文档
    documents = document_loader.load_directory(str(documents_dir))
    if not documents:
        print("未能成功加载任何文档")
        return
        
    # 将文档添加到向量存储
    vector_store.add_documents(documents)
    print(f"成功加载 {len(documents)} 个文档到知识库")

def main():
    """主函数"""
    # 创建必要的目录结构
    setup_directories()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 初始化系统组件
    components = initialize_system()
    
    # 根据命令行参数执行相应的操作
    if args.load_documents:
        load_documents(components)
        
    if args.interactive:
        interactive_mode(components)
        
    if args.capture_mode or args.video_file:
        capture_mode(components, args.video_file)
        
    # 如果没有指定任何操作，默认进入交互式模式
    if not (args.load_documents or args.interactive or args.capture_mode or args.video_file):
        print("\n没有指定操作模式，默认进入交互式问答模式")
        interactive_mode(components)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已中断")
    except Exception as e:
        logger.error(f"程序出错: {str(e)}", exc_info=True)
        print(f"\n程序出错: {str(e)}")
    finally:
        print("\n程序已退出")
