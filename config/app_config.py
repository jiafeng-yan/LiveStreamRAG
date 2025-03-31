import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 应用配置
APP_CONFIG = {
    # 屏幕捕获设置
    "capture": {
        "interval": 1.0,  # 截图间隔(秒)
        "region": None,  # 截图区域(x, y, width, height)，None表示用户手动选择
        "ocr_lang": "chi_sim+eng",  # OCR语言
    },
    # 知识库设置
    "knowledge_base": {
        "db_path": "data/vector_store",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    # LLM设置
    "llm": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        # "model": "deepseek/deepseek-chat",
        "model": "deepseek/deepseek-v3-base:free",
        "temperature": 0.3,
        "max_tokens": 500,
    },
    # RAG设置
    "rag": {
        "top_k": 3,  # 检索文档数量
        "similarity_threshold": 0.6,  # 相似度阈值
    },
}
