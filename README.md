# LiveStreamRAG

<!-- ![image](resources\icon.png) -->

LiveStreamRAG 是一款专为直播场景设计的智能问答系统，旨在提升主播与观众的互动效率。通过微调视觉语言模型（VLM, Qwen-VL-7B），系统能从直播评论区的屏幕截图中精准提取观众提问，过滤噪声评论。结合检索增强生成（RAG）技术，系统从产品知识库中检索相关信息，并由大语言模型（LLM, Qwen3-8B）生成准确、自然的回答。LiveStreamRAG 实现了端到端的自动化流程，包括图像去重、Redis 输出去重和 Windows UI 展示，兼具算法深度与工程价值，为直播电商和内容创作提供高效、实用的 AI 解决方案。


<!-- This project is an AI-powered co-pilot designed for the dynamic world of live streaming. It leverages a fine-tuned Vision Language Model (VLM) to accurately identify viewer questions from the fast-moving chat screen, and uses a Retrieval-Augmented Generation (RAG) engine to instantly provide verified answers from a local knowledge base. By automating Q&A, StreamOracle empowers streamers to focus on creating engaging content, ensuring no important question goes unanswered. -->

<img src="resources\icon.png" width=100 alt="image">

## 功能特点

- 通过 VLM 针对化提取评论区中提问
- 支持多种文档格式的知识库管理
- 基于向量数据库的高效检索
- 支持本地或者 API 调用模型
- 支持 Redis 持久化历史数据

## 安装

```bash
# 克隆仓库
git clone git@github.com:jiafeng-yan/LiveStreamRAG.git LiveStreamRAG-Main
cd LiveStreamRAG-Main

# 初始化环境
conda create -n ls_rag python=3.10
conda activate ls_rag

# 安装依赖
pip install -r requirements.txt
```

## 环境配置

在项目根目录创建`.env`文件，添加以下配置（根据需要调整）：

```
# LLM API配置
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# 向量存储配置
VECTOR_DIR=./data/vector_store

# 日志配置
LOG_DIR=./logs
```

同时可以按需调整项目配置文件 `config/app_config.py` 。

## 运行

### 1. As a Project

```bash
python3 main.py
```

### 2. As a Package

```bash
pip install -e .
# `pip uninstall LiveStreamRAG` to uninstall
```

### 3. As a App

```bash
python3 build_exe.py
# run dist/LiveStreamRAG.exe
```

### Additional Features

```bash
# 可以通过使用 Redis 持久化存储，保证持久化去重
# 开启 Redis 服务
redis-server.exe
# 相关数据存储在 dump.rdb 文件中
```

## 功能集成

### 知识库管理

```python
from knowledge_base.document_loader import DocumentLoader
from knowledge_base.vectorization import Vectorizer

# 加载文档到知识库
loader = DocumentLoader()
documents = loader.load_directory("path/to/documents")

# 向量化文档
vectorizer = Vectorizer()
vectorizer.vectorize_documents(documents)
```

### 检索和回答生成

```python
from rag_engine.generation import RAGEngine

# 初始化RAG引擎
rag_engine = RAGEngine()

# 生成回答
query = "产品有哪些功能特点？"
response = await rag_engine.generate_response(query)
print(response)
```

### 屏幕捕获和 OCR

```python
from screen_capture.capture import ScreenCapture
from screen_capture.ocr import OCRProcessor

# 初始化屏幕捕获和OCR处理器
screen_capture = ScreenCapture()
ocr_processor = OCRProcessor()

# 配置捕获区域
screen_capture.set_capture_region((360, 1200, 500, 450))

# 捕获并处理评论
image = screen_capture.capture()
comments = ocr_processor.process_image(image)
print(comments)
```

## 项目结构

```
llm_rag/
├── chat_response/             # 回复生成和格式化
├── knowledge_base/            # 知识库管理
├── llm_interface/             # 语言模型接口
├── rag_engine/                # 检索增强生成引擎
├── screen_capture/            # 屏幕捕获和OCR
├── tests/                     # 测试代码
├── utils/                     # 工具函数
├── data/                      # 数据目录
│   ├── documents/             # 文档存储
│   ├── vector_store/          # 向量存储
│   ├── video/                 # 测试视频
│   └── debug_vlm/             # 调试图像
├── logs/                      # 日志目录
├── .env                       # 环境配置
├── requirements.txt           # 依赖列表
├── setup.py                   # 安装配置
└── README.md                  # 说明文档
```

## 许可证

MIT

## 贡献

欢迎提交问题和拉取请求。
