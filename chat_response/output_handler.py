import os
import datetime
import json
from config.app_config import APP_CONFIG


class OutputHandler:
    """输出处理器，用于打印和记录交互"""

    def __init__(self, log_dir=None):
        """
        初始化输出处理器
        
        Args:
            log_dir: 日志保存目录，默认使用配置中的设置
        """
        self.log_dir = log_dir if log_dir else APP_CONFIG["output"]["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        self.log_file = os.path.join(self.log_dir, f"interactions_{current_date}.log")
        
        # 打印初始化信息
        print(f"日志将保存到: {self.log_file}")

    def print_response(self, query, response):
        """
        打印用户查询和系统回复
        
        Args:
            query: 用户查询
            response: 系统回复
        """
        print("\n" + "="*50)
        print(f"问题: {query}")
        print("-"*50)
        print(f"回答: {response}")
        print("="*50)

    def log_interaction(self, query, response):
        """
        记录交互到日志文件
        
        Args:
            query: 用户查询
            response: 系统回复
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = {
                "timestamp": timestamp,
                "query": query,
                "response": response
            }
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"记录交互时出错: {e}")
