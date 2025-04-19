import os
import subprocess
import shutil
import sys

def build_executable():
    print("开始构建可执行文件...")
    
    # 检查PyInstaller是否已安装
    try:
        import PyInstaller
        print("PyInstaller已安装")
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller安装完成")
    
    # 创建dist目录（如果不存在）
    if not os.path.exists("dist"):
        os.makedirs("dist")
    
    # 清理旧的构建文件
    for folder in ["build", "dist/LLM_RAG"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"已清理 {folder} 文件夹")
    
    # 构建可执行文件
    cmd = [
        "pyinstaller",
        "--name=LLM_RAG",
        "--windowed",  # 无控制台窗口
        "--onedir",    # 创建一个目录包含可执行文件
        "--clean",     # 清理PyInstaller缓存
        "--add-data=config;config",  # 添加配置文件
        "ui/app.py"    # 入口脚本
    ]
    
    # 对于Windows系统，使用分号分隔路径
    if sys.platform.startswith("win"):
        cmd[5] = "--add-data=config;config"
    else:  # macOS和Linux使用冒号
        cmd[5] = "--add-data=config:config"
    
    # 执行PyInstaller命令
    subprocess.check_call(cmd)
    print("可执行文件构建完成！可在 dist/LLM_RAG 目录中找到")

if __name__ == "__main__":
    build_executable() 