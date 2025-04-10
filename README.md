# LiveStreamRAG

一个创新的人工智能应用，结合大语言模型(LLM)和检索增强生成技术(RAG)，专为直播场景设计。该系统能实时捕获直播评论区内容，通过光学字符识别(OCR)提取文本，检索相关产品知识，并生成专业、准确的回应。

## 核心价值

- 🚀 **提升互动效率**：自动识别并回应观众问题，减轻主播负担
- 🎯 **专业知识支持**：基于产品文档提供权威回答，避免误导信息
- 📊 **智能优先级**：可配置回应策略，优先处理重要评论
- 🔄 **实时性能**：低延迟响应系统，确保互动流畅性
- 🛠️ **高度可定制**：适应不同产品、行业和直播风格

## 功能特点

- **实时屏幕 OCR**: 从直播聊天区捕获并识别文字评论
- **知识库管理**: 支持多种格式文档的加载、向量化和检索
- **RAG 引擎**: 结合检索到的产品知识，生成准确的回应
- **测试模式**: 支持模拟评论输入进行系统测试

## 技术架构

- **屏幕捕获**: PyAutoGUI, OpenCV
<!-- - **OCR 识别**: Tesseract OCR -->
- **OCR 识别**: PaddleOCR
- **知识库**: LangChain, ChromaDB, Sentence Transformers
- **LLM 接口**: OpenRouter (deepseek-chat)

## 安装与配置

### 环境准备

```bash
# 创建并激活conda环境
conda create -n rag python=3.11
conda activate rag

# 安装依赖
pip install -r requirements.txt
```

<!-- ### Tesseract OCR 安装

根据操作系统安装 Tesseract OCR:

- **Windows**: 从[GitHub](https://github.com/UB-Mannheim/tesseract/wiki)下载安装
- **MacOS**: `brew install tesseract tesseract-lang`
- **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim` -->

### PaddleOCR 安装
首先安装对应版本 PaddlePaddle 及其依赖项，
其次安装 PaddleOCR，
运行过程中可能与高版本 Torch 存在不适配现象，具体可以参考 [Issue14904](https://github.com/PaddlePaddle/PaddleOCR/issues/14904) 进行解决。

### OpenRouter API 配置

在项目根目录创建`.env`文件:

```
OPENROUTER_API_KEY=your_api_key_here
```

## 使用方法

1. 将产品文档放入`data/documents`目录
1.1. 运行 Redis 服务:
   ```bash
   cd path/to/redis     ## "E:/Redis/Redis-x64-5.0.14.1"
   redis-server.exe redis.windows.conf
   ```
2. 运行主程序:
   ```bash
   python main.py
   ```
3. 或者运行测试模式:
   ```bash
   python tests/test_rag_system.py
   ```

## 目录结构

```
├── screen_capture/ - 屏幕捕获模块
├── knowledge_base/ - 知识库管理
├── llm_interface/ - 大语言模型接口
├── rag_engine/ - RAG核心逻辑
├── chat_response/ - 回复处理
├── config/ - 配置文件
├── tests/ - 测试模块
├── data/ - 数据和日志
├── main.py - 主程序入口
└── requirements.txt - 依赖包
```
