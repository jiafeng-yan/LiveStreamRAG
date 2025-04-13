# LiveStreamRAG

一个创新的人工智能应用，结合大语言模型(LLM)和检索增强生成技术(RAG)，专为直播场景设计。该系统能实时捕获直播评论区内容，通过视觉语言模型(VLM)提取文本，检索相关产品知识，并生成专业、准确的回应。

## 核心价值

- 🚀 **提升互动效率**：自动识别并回应观众问题，减轻主播负担
- 🎯 **专业知识支持**：基于产品文档提供权威回答，避免误导信息
- 📊 **智能优先级**：可配置回应策略，优先识别提问内容
- 🔄 **实时性能**：低延迟响应系统，确保互动流畅性
- 🛠️ **高度可定制**：适应不同产品、行业和直播风格

## 功能特点

- **实时屏幕捕获**: 从直播聊天区捕获视觉内容
- **VLM 评论识别**: 利用视觉大模型智能提取评论中的问题
- **智能问句去重**: 通过精确匹配、Redis 持久化和语义相似度多重去重
- **知识库管理**: 支持多种格式文档的加载、向量化和检索
- **RAG 引擎**: 结合检索到的产品知识，生成准确的回应
- **多模型轮换**: 支持自动切换多个 VLM 模型，避免单一模型限制
- **测试模式**: 支持模拟评论输入及视频文件测试

## 技术架构

- **屏幕捕获**: PyAutoGUI, OpenCV
- **评论识别**: OpenRouter VLM 模型 (Qwen2.5-VL, GPT-4V, Gemini Pro Vision 等)
- **问句去重**: Redis 持久化 + Sentence Transformers 语义相似度
- **知识库**: LangChain, ChromaDB, Sentence Transformers
- **LLM 接口**: OpenRouter API (多种大语言模型)
- **图像处理**: OpenCV, PIL

## 安装与配置

### 环境准备

```bash
# 创建并激活conda环境
conda create -n rag python=3.11
conda activate rag

# 安装依赖
pip install -r requirements.txt
```

### 配置文件设置

系统的所有可配置参数都位于`config/app_config.py`文件中，主要包括：

1. **屏幕捕获设置**: 截图间隔、捕获区域等
2. **知识库设置**: 嵌入模型、向量存储路径等
3. **LLM 设置**: API 密钥、模型选择、生成参数
4. **OCR/VLM 设置**: 模型配置、提示词、API 参数
5. **去重设置**: Redis 使用、语义相似度阈值等
6. **输出设置**: 日志存储、响应格式化参数

### 环境变量配置

在项目根目录创建`.env`文件:

```
OPENROUTER_API_KEY=your_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password  # 如不需要密码则删除此行
SENTENCE_TRANSFORMER_MODEL=paraphrase-multilingual-MiniLM-L12-v2  # 可选，指定语义模型
```

### Redis 安装（可选）

如果需要使用 Redis 持久化功能，请安装并配置 Redis:

1. **Windows**:

   - 下载 Redis Windows 版本: https://github.com/microsoftarchive/redis/releases
   - 运行`redis-server.exe`启动 Redis 服务

2. **Linux**:

   ```bash
   sudo apt update
   sudo apt install redis-server
   sudo systemctl start redis-server
   ```

3. **Mac**:
   ```bash
   brew install redis
   brew services start redis
   ```

## 问句去重机制

系统使用三层去重机制确保不回答重复问题:

1. **精确匹配**: 使用内存集合过滤完全相同的问题
2. **Redis 持久化**: 使用 Redis 存储历史问题，确保程序重启后仍能避免回答重复问题
3. **语义相似度**: 使用 Sentence Transformers 计算问句相似度，过滤表达不同但含义相同的问题

可以通过`config/app_config.py`调整去重配置:

```python
"deduplication": {
    "use_redis": True,           # 是否使用Redis缓存
    "use_semantic": True,        # 是否使用语义相似度计算
    "similarity_threshold": 0.85, # 语义相似度阈值(0-1)
    "max_cache_items": 2000,     # Redis中最多存储的历史评论数
}
```

## 性能优化

为提高系统在长时间运行时的稳定性和效率，LiveStreamRAG 实现了多项性能优化功能：

1. **跳帧处理**: 系统可配置跳过特定数量的帧，减少 API 调用频率，避免 API 使用限制

   ```python
   "performance_optimization": {
       "skip_frames": 5,  # 每处理1帧后跳过5帧
   }
   ```

2. **运动检测**: 通过比较前后帧的差异，只在画面发生实质变化时进行 OCR 处理

   ```python
   "performance_optimization": {
       "enable_motion_detection": True,  # 启用运动检测
       "motion_threshold": 20,  # 变化检测阈值，值越大需要越明显的变化
   }
   ```

3. **捕获间隔优化**: 根据需要调整屏幕捕获的时间间隔

   ```python
   "capture": {
       "interval": 2.0,  # 捕获间隔(秒)
   }
   ```

4. **模型轮换机制**: 当一个 VLM 模型达到使用限制时，自动切换到下一个可用模型

这些优化选项可以在`config/app_config.py`文件中根据实际需求进行调整。

## 使用方法

1. 如果启用了 Redis（可选）:

   ```bash
   # Windows
   redis-server.exe

   # Linux/Mac
   redis-server
   ```

2. 将产品文档放入`data/documents`目录

3. 运行主程序:

   ```bash
   python main.py
   ```

4. 或者运行测试模式:

   ```bash
   # 测试RAG系统
   python tests/test_rag_system.py

   # 测试VLM评论识别 (使用视频文件)
   python tests/test_ocr_vlm.py
   ```

## 目录结构

```
├── screen_capture/ - 屏幕捕获与评论提取模块
│   ├── capture.py - 屏幕捕获功能
│   └── ocr.py - 基于VLM的评论识别
├── knowledge_base/ - 知识库管理
│   ├── document_loader.py - 文档加载
│   ├── embedding.py - 嵌入模型
│   └── vector_store.py - 向量存储
├── llm_interface/ - 大语言模型接口
│   ├── llm_client.py - LLM客户端
│   └── prompt_templates.py - 提示词模板
├── rag_engine/ - RAG核心逻辑
│   ├── generation.py - 回答生成
│   └── retrieval.py - 知识检索
├── chat_response/ - 回复处理
│   ├── output_handler.py - 输出处理
│   └── response_formatter.py - 回复格式化
├── config/ - 配置文件
│   └── app_config.py - 应用配置
├── tests/ - 测试模块
│   ├── test_rag_system.py - RAG系统测试
│   └── test_ocr_vlm.py - VLM评论识别测试
├── data/ - 数据和日志目录
├── main.py - 主程序入口
└── requirements.txt - 依赖包
```

## 开发计划

以下是项目后续开发计划：

- [x] Redis 缓存: 添加 Redis 支持以缓存频繁查询的问题
- [x] 问句去重: 通过 Redis 持久化和语义相似度优化问题去重逻辑
- [ ] 多线程处理: 实现多线程以提高并发性能
- [ ] 整机测试: 进行完整系统的稳定性测试
- [ ] 模拟测试: 创建更多模拟测试场景
