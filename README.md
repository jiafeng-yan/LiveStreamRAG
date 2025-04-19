# LLM RAG 系统

基于大型语言模型的检索增强生成系统，支持知识库管理、文档检索和自动回复功能。

## 功能特点

- 支持多种文档格式的知识库管理
- 基于向量数据库的高效检索
- 集成多种大型语言模型（LLM）
- 屏幕捕获和 OCR 功能，支持提取直播评论
- 多模态视觉语言模型（VLM）集成

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/llm_rag.git
cd llm_rag

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
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

## 使用方法

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
