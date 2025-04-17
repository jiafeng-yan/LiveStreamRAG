import asyncio
import os
import traceback
from pathlib import Path
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
import cv2
import numpy as np

class LiveStreamRAGSystem:
    """直播RAG系统类"""

    def __init__(self):
        """初始化系统环境"""
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
        print("初始化系统环境...")

        try:
            # 初始化知识库
            print("初始化知识库...")
            embedding_model = EmbeddingModel(APP_CONFIG["knowledge_base"]["embedding_model"])
            print(f"嵌入模型初始化成功")

            self.vector_store = VectorStore(
                embedding_model=embedding_model,
                db_path=APP_CONFIG["knowledge_base"]["db_path"],
                chunk_size=APP_CONFIG["knowledge_base"]["chunk_size"],
                chunk_overlap=APP_CONFIG["knowledge_base"]["chunk_overlap"],
                embedding_model_name=APP_CONFIG["knowledge_base"]["embedding_model"],
            )
            print("向量存储初始化成功")

            # 初始化LLM客户端
            print(f"正在初始化LLM客户端，模型: {APP_CONFIG['llm']['model']}")
            self.llm_client = OpenRouterClient(
                api_key=APP_CONFIG["llm"]["api_key"], 
                model=APP_CONFIG["llm"]["model"],
                temperature=APP_CONFIG["llm"]["temperature"],
                max_tokens=APP_CONFIG["llm"]["max_tokens"]
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

            # 初始化屏幕捕获器
            self.screen_capture = ScreenCapture(APP_CONFIG["capture"]["region"])
            print("屏幕捕获器初始化成功")

            # 初始化OCR处理器，使用增强去重功能
            try:
                self.ocr_processor = OCRProcessor(
                    use_redis=APP_CONFIG["deduplication"]["use_redis"],
                    use_semantic=APP_CONFIG["deduplication"]["use_semantic"],
                    similarity_threshold=APP_CONFIG["deduplication"]["similarity_threshold"]
                )
                print("OCR处理器初始化成功")
            except Exception as e:
                print(f"初始化OCR处理器时出错: {e}")
                print("将使用默认OCR处理器（无语义相似度功能）...")
                # 使用没有语义相似度功能的OCR处理器
                self.ocr_processor = OCRProcessor(use_redis=False, use_semantic=False)
                print("默认OCR处理器初始化成功")

        except Exception as e:
            print(f"初始化组件时出错: {str(e)}")
            traceback.print_exc()
            raise

        print("系统环境初始化完成")

    async def load_knowledge_base(self, data_dir="data/documents"):
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
            self.vector_store.add_documents(documents, "initial_docs")
            print(f"已将 {len(documents)} 个文档添加到知识库")
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
            truncated_response = ResponseFormatter.truncate_if_needed(formatted_response)

            # 输出和记录
            self.output_handler.print_response(comment, truncated_response)
            self.output_handler.log_interaction(comment, truncated_response)

            return truncated_response

        except Exception as e:
            error_msg = f"处理评论时出错: {e}"
            print(error_msg)
            return error_msg

    async def run(self):
        """运行系统主循环"""
        print("系统启动，开始监控直播评论...")

        try:
            debug_mode = APP_CONFIG["capture"]["debug_mode"]
            debug_dir = APP_CONFIG["capture"]["debug_dir"]
            
            if debug_mode:
                os.makedirs(debug_dir, exist_ok=True)
                print(f"调试模式已启用，截图将保存到 {debug_dir}")
            
            # 性能优化参数
            frame_counter = 0
            skip_frames = APP_CONFIG["ocr"]["performance_optimization"]["skip_frames"]
            enable_motion_detection = APP_CONFIG["ocr"]["performance_optimization"]["enable_motion_detection"]
            motion_threshold = APP_CONFIG["ocr"]["performance_optimization"]["motion_threshold"]
            previous_frame = None
            
            print(f"性能优化: 跳帧={skip_frames}, 运动检测={'启用' if enable_motion_detection else '禁用'}")
            
            while True:
                # 捕获屏幕
                screenshot_path = self.screen_capture.capture_and_save()
                
                # 在调试模式下保存截图副本
                if debug_mode:
                    import shutil
                    debug_path = os.path.join(debug_dir, os.path.basename(screenshot_path))
                    shutil.copy2(screenshot_path, debug_path)
                    print(f"已保存调试截图: {debug_path}")
                
                # 性能优化: 跳帧处理
                frame_counter += 1
                should_process = (frame_counter % (skip_frames + 1) == 0)
                
                # 性能优化: 运动检测
                if enable_motion_detection and should_process:
                    current_frame = cv2.imread(screenshot_path)
                    if previous_frame is not None:
                        # 计算两帧差异
                        diff = cv2.absdiff(current_frame, previous_frame)
                        non_zero_count = np.count_nonzero(diff)
                        avg_diff = non_zero_count / diff.size
                        should_process = avg_diff > (motion_threshold / 1000.0)
                        if debug_mode and not should_process:
                            print(f"跳过处理: 图像变化量({avg_diff:.5f})低于阈值({motion_threshold/1000.0:.5f})")
                    previous_frame = current_frame
                
                if should_process:
                    # 提取评论
                    comments = await self.ocr_processor.process_image(screenshot_path)

                    # 处理新评论
                    if comments:
                        print(f"检测到 {len(comments)} 条新评论")
                        for comment in comments:
                            await self.process_comment(comment)
                    else:
                        print("未检测到新评论")
                
                # 删除截图文件
                os.remove(screenshot_path)

                # 等待下一次捕获
                await asyncio.sleep(APP_CONFIG["capture"]["interval"])

        except KeyboardInterrupt:
            print("\n程序已停止")
        except Exception as e:
            print(f"运行时出错: {str(e)}")
            traceback.print_exc()


async def main():
    """主函数"""
    # 确保目录存在
    os.makedirs("data/screenshots", exist_ok=True)
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs(APP_CONFIG["output"]["log_dir"], exist_ok=True)

    print("启动RAG直播评论系统...")
    print("请将产品文档放入 data/documents 目录")
    print("按Ctrl+C停止程序")

    # 创建并运行系统
    system = LiveStreamRAGSystem()
    await system.load_knowledge_base()
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
