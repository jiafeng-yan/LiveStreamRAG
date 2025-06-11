import sys
import os
import asyncio
import time
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QCheckBox, QGroupBox, QMessageBox, QTextEdit,
    QProgressBar, QStatusBar, QLineEdit, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QScreen, QGuiApplication, QCursor

# 导入本地模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.app_config import APP_CONFIG

# 配置文件路径
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "user_config.json")


class ScreenshotOverlay(QWidget):
    """屏幕截图覆盖层，用于选择屏幕区域"""
    
    region_selected = pyqtSignal(tuple)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        
        # 设置透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 截图初始化
        self.begin = QPoint()
        self.end = QPoint()
        self.is_drawing = False
        
        # 抓取整个屏幕截图作为背景
        self.screenshot = QGuiApplication.primaryScreen().grabWindow(0)
        
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.screenshot)
        
        # 绘制半透明遮罩
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        if self.is_drawing:
            # 绘制选区矩形
            rect = QRect(self.begin, self.end)
            painter.setPen(QPen(QColor(255, 255, 0), 3))
            painter.drawRect(rect)
            
            # 绘制选区内容（无遮罩）
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            
            # 绘制选区边框
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(255, 255, 0), 3))
            painter.drawRect(rect)
            
            # 显示选区尺寸信息
            size_text = f"{abs(rect.width())} x {abs(rect.height())}"
            painter.drawText(rect.x(), rect.y() - 10, size_text)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin = event.pos()
            self.end = event.pos()
            self.is_drawing = True
            self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.is_drawing:
            self.end = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.hide()
            
            # 计算选区的坐标和尺寸
            x = min(self.begin.x(), self.end.x())
            y = min(self.begin.y(), self.end.y())
            width = abs(self.begin.x() - self.end.x())
            height = abs(self.begin.y() - self.end.y())
            
            # 发送信号，包含选区信息
            self.region_selected.emit((x, y, width, height))
            
            # 关闭窗口
            self.close()
            self.deleteLater()
    
    def keyPressEvent(self, event):
        """键盘按下事件"""
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            # 关闭窗口并通知主窗口
            self.close()
            self.deleteLater()
            

class PointPickerOverlay(QWidget):
    """点选择覆盖层，用于选择聊天框发送位置"""
    
    point_selected = pyqtSignal(QPoint)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        
        # 设置透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 抓取整个屏幕截图作为背景
        self.screenshot = QGuiApplication.primaryScreen().grabWindow(0)
        
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.screenshot)
        
        # 绘制半透明遮罩
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
        
        # 绘制提示文本
        font = painter.font()
        font.setPointSize(20)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 0))
        
        text = "请点击聊天框中的发送区域"
        rect = self.rect()
        painter.drawText(
            (rect.width() - painter.fontMetrics().horizontalAdvance(text)) // 2,
            rect.height() // 2,
            text
        )
        
        # 获取鼠标位置并绘制十字光标
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawLine(cursor_pos.x() - 10, cursor_pos.y(), cursor_pos.x() + 10, cursor_pos.y())
        painter.drawLine(cursor_pos.x(), cursor_pos.y() - 10, cursor_pos.x(), cursor_pos.y() + 10)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.hide()
            # 发送所选点的位置
            self.point_selected.emit(event.pos())
            # 关闭窗口
            self.close()
            self.deleteLater()
    
    def keyPressEvent(self, event):
        """键盘按下事件"""
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            # 关闭窗口并通知主窗口
            self.close()
            self.deleteLater()


class WorkerThread(QThread):
    """工作线程，用于运行异步任务"""
    
    update_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    comment_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, system_instance):
        super().__init__()
        self.system = system_instance
        self.running = False
    
    def run(self):
        """运行线程"""
        self.running = True
        
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 运行异步任务
            self.update_signal.emit("初始化知识库...")
            loop.run_until_complete(self.system.load_knowledge_base())
            
            self.update_signal.emit("系统已启动，开始监控直播评论...")
            loop.run_until_complete(self.run_system())
        except Exception as e:
            self.error_signal.emit(f"运行时出错: {str(e)}")
        finally:
            loop.close()
            self.finished_signal.emit()
    
    async def run_system(self):
        """运行系统主循环"""
        from screen_capture.capture import ScreenCapture
        import cv2
        import numpy as np
        import pyautogui
        
        # 初始化计数器和性能参数
        frame_counter = 0
        skip_frames = APP_CONFIG["ocr"]["performance_optimization"]["skip_frames"]
        enable_motion_detection = APP_CONFIG["ocr"]["performance_optimization"]["enable_motion_detection"]
        motion_threshold = APP_CONFIG["ocr"]["performance_optimization"]["motion_threshold"]
        previous_frame = None
        
        self.log_signal.emit(f"性能优化: 跳帧={skip_frames}, 运动检测={'启用' if enable_motion_detection else '禁用'}")
        
        while self.running:
            try:
                # 捕获屏幕
                screenshot_path = self.system.screen_capture.capture_and_save()
                
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
                        self.log_signal.emit(f"图像变化量: {avg_diff:.5f}, 阈值: {motion_threshold/1000.0:.5f}")
                    previous_frame = current_frame
                
                if should_process:
                    # 提取评论
                    self.update_signal.emit("正在处理截图...")
                    comments = await self.system.ocr_processor.process_image(screenshot_path)

                    # 处理新评论
                    if comments:
                        self.log_signal.emit(f"检测到 {len(comments)} 条新评论")
                        for comment in comments:
                            if not self.running:
                                break
                                
                            self.comment_signal.emit(comment)
                            response = await self.system.process_comment(comment)
                            
                            # 如果设置了聊天框位置，将回复复制到聊天框
                            if hasattr(self, 'chat_input_pos') and self.chat_input_pos:
                                self.copy_to_chat(response)
                    else:
                        self.log_signal.emit("未检测到新评论")
                
                # 删除截图文件
                os.remove(screenshot_path)

                # 等待下一次捕获
                await asyncio.sleep(APP_CONFIG["capture"]["interval"])
            except Exception as e:
                self.error_signal.emit(f"循环中出错: {str(e)}")
                await asyncio.sleep(1)  # 避免错误循环过快
    
    def stop(self):
        """停止线程"""
        self.running = False
    
    def set_chat_input_position(self, pos):
        """设置聊天框输入位置"""
        self.chat_input_pos = pos
    
    def copy_to_chat(self, text):
        """将文本复制到聊天框并发送"""
        try:
            import pyperclip
            import pyautogui
            
            # 复制文本到剪贴板
            pyperclip.copy(text)
            
            # 移动到聊天框位置并点击
            pyautogui.click(self.chat_input_pos.x(), self.chat_input_pos.y())
            time.sleep(0.1)
            
            # 粘贴文本
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.1)
            
            # 发送消息（通常按Enter键）
            pyautogui.press('enter')
            
            self.log_signal.emit("已复制回复到聊天框并发送")
        except Exception as e:
            self.error_signal.emit(f"复制到聊天框时出错: {str(e)}")


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 窗口设置
        self.setWindowTitle("LiveStreamRAG直播助手")
        self.setMinimumSize(800, 600)
        
        # 加载用户配置
        self.user_config = self.load_user_config()
        
        # 界面初始化
        self.init_ui()
        
        # 系统实例
        self.system = None
        self.worker_thread = None
        
        # 区域选择覆盖层
        self.region_overlay = None
        self.point_overlay = None
        
        # 捕获区域和聊天框位置
        self.capture_region = None
        self.chat_input_pos = None
    
    def load_user_config(self):
        """加载用户配置"""
        default_config = {
            "document_path": APP_CONFIG["rag"]["document_path"]
        }
        
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    # 更新默认配置
                    default_config.update(loaded_config)
                    self.add_log(f"已加载用户配置")
            except Exception as e:
                self.add_log(f"加载用户配置失败: {str(e)}")
        
        # 确保配置中的目录存在
        os.makedirs(default_config["document_path"], exist_ok=True)
        
        return default_config
    
    def save_user_config(self):
        """保存用户配置"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
            
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.user_config, f, ensure_ascii=False, indent=4)
            
            self.add_log(f"用户配置已保存")
        except Exception as e:
            self.add_log(f"保存用户配置失败: {str(e)}")
    
    def init_ui(self):
        """初始化界面"""
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 顶部配置区域
        config_group = QGroupBox("配置")
        config_layout = QVBoxLayout(config_group)
        
        # 区域选择
        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("捕获区域:"))
        self.region_label = QLabel("未选择")
        region_layout.addWidget(self.region_label)
        self.select_region_btn = QPushButton("选择区域")
        self.select_region_btn.clicked.connect(self.select_region)
        region_layout.addWidget(self.select_region_btn)
        config_layout.addLayout(region_layout)
        
        # 聊天框选择
        chat_layout = QHBoxLayout()
        chat_layout.addWidget(QLabel("聊天框位置:"))
        self.chat_pos_label = QLabel("未选择")
        chat_layout.addWidget(self.chat_pos_label)
        self.select_chat_btn = QPushButton("选择位置")
        self.select_chat_btn.clicked.connect(self.select_chat_position)
        chat_layout.addWidget(self.select_chat_btn)
        config_layout.addLayout(chat_layout)
        
        # 文档目录设置
        doc_path_layout = QHBoxLayout()
        doc_path_layout.addWidget(QLabel("文档目录:"))
        self.doc_path_edit = QLineEdit(self.user_config["document_path"])
        doc_path_layout.addWidget(self.doc_path_edit)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_doc_path)
        doc_path_layout.addWidget(self.browse_btn)
        config_layout.addLayout(doc_path_layout)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动系统")
        self.start_btn.clicked.connect(self.start_system)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止系统")
        self.stop_btn.clicked.connect(self.stop_system)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        config_layout.addLayout(control_layout)
        
        main_layout.addWidget(config_group)
        
        # 日志区域
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_group)
        
        # 评论和回复区域
        interaction_layout = QHBoxLayout()
        
        # 评论区
        comments_group = QGroupBox("检测到的评论")
        comments_layout = QVBoxLayout(comments_group)
        self.comments_text = QTextEdit()
        self.comments_text.setReadOnly(True)
        comments_layout.addWidget(self.comments_text)
        interaction_layout.addWidget(comments_group)
        
        # 回复区
        responses_group = QGroupBox("系统回复")
        responses_layout = QVBoxLayout(responses_group)
        self.responses_text = QTextEdit()
        self.responses_text.setReadOnly(True)
        responses_layout.addWidget(self.responses_text)
        interaction_layout.addWidget(responses_group)
        
        main_layout.addLayout(interaction_layout)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("系统未启动")
        self.status_bar.addWidget(self.status_label)
        
        # 设置主部件
        self.setCentralWidget(main_widget)
        
        # 初始日志
        self.add_log("欢迎使用LiveStreamRAG直播助手，请进行配置后启动系统")
    
    def select_region(self):
        """选择捕获区域"""
        # 隐藏主窗口
        self.hide()
        # 短暂延迟以确保窗口完全隐藏
        QApplication.processEvents()
        
        self.region_overlay = ScreenshotOverlay()
        self.region_overlay.region_selected.connect(self.set_region)
        # 当用户按ESC键取消选择时也要显示主窗口
        self.region_overlay.destroyed.connect(self.show)
        self.region_overlay.show()
    
    def set_region(self, region):
        """设置捕获区域"""
        self.capture_region = region
        self.region_label.setText(f"已选择: {region[0]}, {region[1]}, {region[2]} x {region[3]}")
        self.add_log(f"已设置捕获区域: {region}")
        # 显示主窗口
        self.show()
    
    def select_chat_position(self):
        """选择聊天框位置"""
        # 隐藏主窗口
        self.hide()
        # 短暂延迟以确保窗口完全隐藏
        QApplication.processEvents()
        
        self.point_overlay = PointPickerOverlay()
        self.point_overlay.point_selected.connect(self.set_chat_position)
        # 当用户按ESC键取消选择时也要显示主窗口
        self.point_overlay.destroyed.connect(self.show)
        self.point_overlay.show()
    
    def set_chat_position(self, pos):
        """设置聊天框位置"""
        self.chat_input_pos = pos
        self.chat_pos_label.setText(f"已选择: {pos.x()}, {pos.y()}")
        self.add_log(f"已设置聊天框位置: ({pos.x()}, {pos.y()})")
        
        # 显示主窗口
        self.show()
        
        # 如果线程已经在运行，更新聊天框位置
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.set_chat_input_position(pos)
    
    def browse_doc_path(self):
        """浏览并选择文档目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择文档目录", 
            self.doc_path_edit.text(),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if dir_path:
            self.doc_path_edit.setText(dir_path)
            self.user_config["document_path"] = dir_path
            self.save_user_config()
            self.add_log(f"已设置文档目录: {dir_path}")
    
    def start_system(self):
        """启动系统"""
        if not self.capture_region:
            QMessageBox.warning(self, "警告", "请先选择捕获区域")
            return
        
        # 保存当前文档路径设置
        current_path = self.doc_path_edit.text()
        if current_path != self.user_config["document_path"]:
            self.user_config["document_path"] = current_path
            self.save_user_config()
        
        try:
            # 导入必要的模块
            from screen_capture.capture import ScreenCapture
            from screen_capture.ocr import OCRProcessor
            from knowledge_base.embedding import EmbeddingModel
            from knowledge_base.vector_store import VectorStore
            from llm_interface.llm_client import OpenRouterClient
            from rag_engine.retrieval import Retriever
            from rag_engine.generation import RAGEngine
            from chat_response.response_formatter import ResponseFormatter
            from chat_response.output_handler import OutputHandler
            
            # 创建LiveStreamRAGSystem的简化版本
            class UILiveStreamRAGSystem:
                def __init__(self, capture_region, document_path):
                    # 设置捕获区域
                    self.screen_capture = ScreenCapture(capture_region)
                    
                    # 保存文档路径
                    self.document_path = document_path
                    
                    # 初始化知识库相关组件
                    embedding_model = EmbeddingModel(APP_CONFIG["knowledge_base"]["embedding_model"])
                    
                    self.vector_store = VectorStore(
                        embedding_model=embedding_model,
                        db_path=APP_CONFIG["knowledge_base"]["db_path"],
                        chunk_size=APP_CONFIG["knowledge_base"]["chunk_size"],
                        chunk_overlap=APP_CONFIG["knowledge_base"]["chunk_overlap"],
                        embedding_model_name=APP_CONFIG["knowledge_base"]["embedding_model"],
                    )
                    
                    # 初始化LLM客户端
                    self.llm_client = OpenRouterClient(
                        use_local_model=APP_CONFIG['llm']['use_local_model'],
                        model_path=APP_CONFIG['llm']['model_path'],
                        api_key=APP_CONFIG["llm"]["api_key"], 
                        model=APP_CONFIG["llm"]["model"],
                        temperature=APP_CONFIG["llm"]["temperature"],
                        max_tokens=APP_CONFIG["llm"]["max_tokens"]
                    )
                    
                    # 初始化检索器
                    self.retriever = Retriever(
                        self.vector_store,
                        APP_CONFIG["rag"]["top_k"],
                        APP_CONFIG["rag"]["similarity_threshold"],
                    )
                    
                    # 初始化RAG引擎
                    self.rag_engine = RAGEngine(self.retriever, self.llm_client)
                    
                    # 初始化OCR处理器
                    self.ocr_processor = OCRProcessor(
                        use_local_model=APP_CONFIG['ocr']['use_local_model'],
                        use_redis=APP_CONFIG["deduplication"]["use_redis"],
                        use_semantic=APP_CONFIG["deduplication"]["use_semantic"],
                        similarity_threshold=APP_CONFIG["deduplication"]["similarity_threshold"]
                    )
                    
                    # 初始化输出处理器
                    self.output_handler = OutputHandler(use_redis=False, use_semantic=False)
                
                async def load_knowledge_base(self):
                    """加载知识库"""
                    from knowledge_base.document_loader import DocumentLoader
                    import os
                    
                    data_dir = self.document_path
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
                        print(f"警告: 没有在 {data_dir} 目录中找到文档")
                
                async def process_comment(self, comment):
                    """处理单条评论"""
                    try:
                        # 生成回复
                        response = await self.rag_engine.generate_response(comment)
                        
                        # 格式化回复
                        # formatted_response = ResponseFormatter.format_response(response, comment)
                        # truncated_response = ResponseFormatter.truncate_if_needed(formatted_response)
                        truncated_response = response.replace("\n", "\t")
                        
                        # 输出和记录
                        self.output_handler.print_response(comment, truncated_response)
                        self.output_handler.log_interaction(comment, truncated_response)
                        
                        return truncated_response
                    
                    except Exception as e:
                        error_msg = f"处理评论时出错: {e}"
                        print(error_msg)
                        return error_msg
            
            # 创建系统实例
            self.system = UILiveStreamRAGSystem(self.capture_region, self.user_config["document_path"])
            
            # 创建工作线程
            self.worker_thread = WorkerThread(self.system)
            self.worker_thread.update_signal.connect(self.update_status)
            self.worker_thread.log_signal.connect(self.add_log)
            self.worker_thread.comment_signal.connect(self.add_comment)
            self.worker_thread.error_signal.connect(self.show_error)
            self.worker_thread.finished_signal.connect(self.system_finished)
            
            # 如果已经选择了聊天框位置，设置到线程
            if self.chat_input_pos:
                self.worker_thread.set_chat_input_position(self.chat_input_pos)
            
            # 启动线程
            self.worker_thread.start()
            
            # 更新界面
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.select_region_btn.setEnabled(False)
            self.browse_btn.setEnabled(False)
            self.doc_path_edit.setEnabled(False)
            self.update_status("系统启动中...")
            
        except ImportError as e:
            QMessageBox.critical(self, "错误", f"缺少必要的依赖: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动系统时出错: {str(e)}")
    
    def stop_system(self):
        """停止系统"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.update_status("正在停止系统...")
            self.worker_thread.stop()
            self.worker_thread.wait(5000)  # 等待最多5秒
            
            # 如果线程仍在运行，强制终止
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.add_log("系统已强制终止")
            else:
                self.add_log("系统已正常停止")
            
            # 更新界面
            self.system_finished()
    
    def system_finished(self):
        """系统运行结束的处理"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_region_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.doc_path_edit.setEnabled(True)
        self.update_status("系统已停止")
    
    def update_status(self, message):
        """更新状态栏信息"""
        self.status_label.setText(message)
        self.add_log(message)
    
    def add_log(self, message):
        """添加日志"""
        if hasattr(self, 'log_text'):
            self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            self.log_text.ensureCursorVisible()
    
    def add_comment(self, comment):
        """添加评论到界面"""
        self.comments_text.append(f"[{time.strftime('%H:%M:%S')}] {comment}")
        self.comments_text.ensureCursorVisible()
        
        # 监听系统回复
        self.system.output_handler.print_response = self.handle_system_response
    
    def handle_system_response(self, query, response):
        """处理系统回复"""
        self.responses_text.append(f"[{time.strftime('%H:%M:%S')}]\n问题: {query}\n回答: {response}\n")
        self.responses_text.ensureCursorVisible()
    
    def show_error(self, message):
        """显示错误信息"""
        self.add_log(f"错误: {message}")
        QMessageBox.warning(self, "错误", message)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认退出", 
                "系统正在运行，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_system()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    # 确保必要的目录存在
    os.makedirs("data/screenshots", exist_ok=True)
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    
    # 创建应用程序
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 