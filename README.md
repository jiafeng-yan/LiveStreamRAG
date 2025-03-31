# LLM-RAG 直播评论智能回复系统

基于大语言模型的 RAG 应用，从实时屏幕读取直播聊天区评论，并通过大语言模型结合产品知识库生成回应。

## 项目功能

1. **屏幕捕获与 OCR**：自动捕获屏幕指定区域并识别评论文本
2. **知识库管理**：支持多格式文档加载，构建向量化知识库
3. **RAG 检索生成**：检索相关产品知识，结合大模型生成专业回应
4. **实时交互**：对评论进行实时响应

## 系统架构

```
├── screen_capture/ - 屏幕捕获模块
│   ├── capture.py - 截取屏幕区域
│   └── ocr.py - 文字识别处理
├── knowledge_base/ - 知识库管理
│   ├── document_loader.py - 加载产品文档
│   ├── embedding.py - 向量化处理
│   ├── direct_embedding.py - ChromaDB适配器
│   └── vector_store.py - 向量存储
├── llm_interface/ - 大语言模型接口
│   ├── llm_client.py - LLM API调用
│   └── prompt_templates.py - 提示词模板
├── rag_engine/ - RAG核心逻辑
│   ├── retrieval.py - 相关内容检索
│   └── generation.py - 回答生成
├── chat_response/ - 回复处理
│   ├── response_formatter.py - 回复格式化
│   └── output_handler.py - 输出控制
├── config/ - 配置文件
│   └── app_config.py - 应用配置
├── tests/ - 测试模块
│   └── test_rag_system.py - RAG系统测试
├── data/ - 数据目录
│   ├── documents/ - 产品知识文档
│   ├── screenshots/ - 屏幕截图
│   ├── vector_store/ - 向量数据库
│   └── logs/ - 系统日志
├── main.py - 主程序入口
└── requirements.txt - 依赖包
```

## 安装与配置

### 1. 环境配置

```bash
# 创建并激活conda环境
conda create -n rag python=3.10
conda activate rag

# 安装依赖包
pip install -r requirements.txt
```

### 2. Tesseract OCR 安装

#### Windows:

1. 从[Tesseract 官方网站](https://github.com/UB-Mannheim/tesseract/wiki)下载安装程序
2. 安装时选择中文和英文语言包

#### MacOS:

```bash
brew install tesseract
brew install tesseract-lang
```

#### Linux:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-chi-sim
```

### 3. 配置 API 密钥

创建`.env`文件，添加 OpenRouter API 密钥:

```
OPENROUTER_API_KEY=your_api_key_here
```

## 使用方法

### 测试系统

```bash
python tests/test_rag_system.py
```

### 运行主程序

```bash
python main.py
```

1. 首次运行会提示选择直播评论区域
2. 系统会自动捕获该区域的评论并生成回复

### 添加知识库文档

将产品相关文档(PDF、Word、Markdown 等)放入`data/documents`目录

## 技术栈

- **OCR**: Tesseract, OpenCV
- **向量数据库**: ChromaDB
- **嵌入模型**: Sentence Transformers
- **大语言模型**: OpenRouter API (deepseek)
- **文档处理**: PyPDF, python-docx

## 开发说明

- 使用`all-MiniLM-L6-v2`模型进行文本嵌入
- 通过 OpenRouter API 调用 Deepseek 模型进行回复生成
- 支持多种格式的产品文档加载和管理
