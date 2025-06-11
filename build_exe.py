import os
import subprocess
import shutil
import sys
from pathlib import Path

app_name = "LiveStreamRAG"
main_file = "ui/app.py"
icon_path = Path("resources/icon.png")


def build_executable():
    print("开始构建可执行文件...")
    
    # 检查PyInstaller是否已安装
    try:
        import PyInstaller
        print("PyInstaller已安装")
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pyinstaller"])
        print("PyInstaller安装完成")
    
    # 创建dist目录（如果不存在）
    if not os.path.exists("dist"):
        os.makedirs("dist")
    
    # 清理旧的构建文件
    for folder in ["build", f"dist/{app_name}"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"已清理 {folder} 文件夹")
    
    # 构建可执行文件
    cmd = [
        "pyinstaller",
        "--name", app_name,
        "--windowed",  # 无控制台窗口
        "--onedir",    # 创建一个目录包含可执行文件
        "--clean",     # 清理PyInstaller缓存
        "--noconfirm"  # 直接覆盖输出文件，不进行提示
    ]
    
    # 添加图标(如果存在)
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    # 添加数据文件
    # 在Windows上使用分号，在macOS/Linux上使用冒号
    separator = ";" if sys.platform.startswith('win') else ":"
    
    # 添加配置文件
    config_path = Path("config")
    if config_path.exists():
        cmd.extend(["--add-data", f"config{separator}config"])
    
    # 添加资源文件
    resources_path = Path("resources")
    if resources_path.exists():
        cmd.extend(["--add-data", f"resources{separator}resources"])

    if Path('resources/version.txt').exists():
        cmd.extend(["--version-file", "resources/version.txt"])
    
    # 添加其他必要的数据文件
    for data_dir in ["data", "schemas", "templates"]:
        data_path = Path(data_dir)
        if data_path.exists():
            cmd.extend(["--add-data", f"{data_dir}{separator}{data_dir}"])
    
    # 添加入口文件
    cmd.append(main_file)
    
    # 执行PyInstaller命令
    subprocess.check_call(cmd)
    print(f"可执行文件构建完成！可在 dist/{app_name} 目录中找到")

if __name__ == "__main__":
    build_executable()