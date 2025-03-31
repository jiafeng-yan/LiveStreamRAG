# LLM-RAG 直播评论智能助手

基于大语言模型的 RAG（检索增强生成）应用，用于实时识别直播聊天区评论并智能回应。

## 功能特点

- **实时屏幕 OCR**: 从直播聊天区捕获并识别文字评论
- **知识库管理**: 支持多种格式文档的加载、向量化和检索
- **RAG 引擎**: 结合检索到的产品知识，生成准确的回应
- **测试模式**: 支持模拟评论输入进行系统测试

## 技术架构

- **屏幕捕获**: PyAutoGUI, OpenCV
- **OCR 识别**: Tesseract OCR
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

### Tesseract OCR 安装

根据操作系统安装 Tesseract OCR:

- **Windows**: 从[GitHub](https://github.com/UB-Mannheim/tesseract/wiki)下载安装
- **MacOS**: `brew install tesseract tesseract-lang`
- **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim`

### OpenRouter API 配置

在项目根目录创建`.env`文件:

```
OPENROUTER_API_KEY=your_api_key_here
```

## 使用方法

1. 将产品文档放入`data/documents`目录
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
