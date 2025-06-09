import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 应用配置
APP_CONFIG = {
    # 屏幕捕获设置
    "capture": {
        "interval": 2.0,  # 截图间隔(秒)，可根据性能需求调整：较小的值提高响应速度但增加系统负载，较大的值减少系统负载但可能错过部分评论
        "region": None,  # 截图区域(x, y, width, height)，None表示用户手动选择
        "debug_mode": False,  # 是否保存调试截图
        "debug_dir": "data/debug_screenshots",  # 调试截图保存目录
    },
    # 知识库设置
    "knowledge_base": {
        # "embedding_model": "E:/data/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "db_path": "data/vector_store",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    # LLM设置
    "llm": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": 'qwen/qwen2.5-vl-32b-instruct:free',
        "temperature": 0.3,
        "max_tokens": 500,
    },
    # RAG设置
    "rag": {
        "top_k": 3,  # 检索文档数量
        "similarity_threshold": 0.6,  # 相似度阈值
        "document_path": "data/documents",  # 文档保存目录
        "prompt_template": "请根据以下上下文回答问题。\n\n上下文:\n{context}\n\n问题:\n{query}\n\n回答:",  # RAG提示词模板
    },
    # OCR/VLM 设置
    "ocr": {
        'use_local_model': True,
        # 本地 Huggingface VLM 模型配置
        "local_hf_model": {
            "name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "device": "cuda",
            "prompt": "从这张图片中提取所有的评论文本。",
            "max_tokens": 1500
        },
        # API VLM模型配置
        "vlm_models": {
            "qwen/qwen2.5-vl-32b-instruct:free": {
                "prompt": "请从图片中的评论区识别所有询问主播的问题。只返回问句，每个问题占一行。如果评论不是问句则忽略。问句通常以'吗'、'？'、'怎么'、'什么'、'如何'、'为什么'等词语结尾。",
                "format": "qwen",
                "max_tokens": 1000
            },
            "gpt4vision/gpt-4-vision-preview:free": {
                "prompt": "From the Chinese comments in the image, identify and extract only the questions asked to the streamer. Return only the questions, one per line. Ignore non-question comments. Questions typically end with words like '吗', '？', '怎么', '什么', '如何', '为什么'.",
                "format": "gpt4v",
                "max_tokens": 1000
            },
            "google/gemini-pro-vision:free": {
                "prompt": "Extract only the questions asked to the streamer from the Chinese comments in this image. Return only the questions, one per line. Questions typically end with '吗', '？', '怎么', '什么', '如何', '为什么'. Ignore all non-question comments.",
                "format": "gemini",
                "max_tokens": 1000
            }
        },
        "max_retries": 3,  # 模型调用最大重试次数
        "api_url": "https://openrouter.ai/api/v1/chat/completions",  # OpenRouter API URL
        "temperature": 0.1,  # 模型温度参数
        "performance_optimization": {
            "skip_frames": 5,  # 处理过程中跳过的帧数，增大此值可减少API调用频率
            "min_text_diff": 30,  # 图像内容变化检测阈值，低于此值不进行OCR处理
            "enable_motion_detection": True,  # 启用运动检测，只在检测到图像变化时处理
            "motion_threshold": 20,  # 运动检测阈值
        },
    },
    # 问题去重设置
    "deduplication": {
        "use_redis": False,  # 是否使用Redis缓存
        "use_semantic": True,  # 是否使用语义相似度计算
        "similarity_threshold": 0.85,  # 语义相似度阈值
        "max_cache_items": 2000,  # Redis中最多存储的历史评论数
        "local_cache_size": 1000,  # 本地内存缓存大小
    },
    # Redis设置
    "redis": {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": int(os.getenv("REDIS_DB", 0)),
        "password": os.getenv("REDIS_PASSWORD", None),
        "prefix": "ocr:comment:",  # Redis键前缀
        "ttl": 60*60*24*30,  # 缓存过期时间(秒)，默认30天
    },
    # 语义模型设置
    "semantic": {
        "model": os.getenv("SENTENCE_TRANSFORMER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
    },
    # 输出设置
    "output": {
        "log_dir": "data/logs",  # 日志保存目录
        "max_response_length": 2000,  # 最大回复长度
        "min_keep_ratio": 0.8,  # 截断时保留的最小比例
    }
}
