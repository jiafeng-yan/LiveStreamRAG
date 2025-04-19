#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

def check_pyinstaller():
    """检查是否安装了PyInstaller"""
    try:
        import PyInstaller
        print(f"✅ 已安装PyInstaller {PyInstaller.__version__}")
        return True
    except ImportError:
        print("❌ 未安装PyInstaller，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✅ PyInstaller安装成功")
            return True
        except Exception as e:
            print(f"❌ 安装PyInstaller失败: {str(e)}")
            print("请手动运行: pip install pyinstaller")
            return False

def create_dist_dir():
    """创建并清理dist目录"""
    dist_dir = Path("dist")
    if not dist_dir.exists():
        dist_dir.mkdir()
        print(f"✅ 已创建dist目录")
    
    # 清理旧的构建文件
    build_dir = Path("build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"✅ 已清理build目录")
    
    # 检查并清理旧的spec文件
    spec_file = Path("LLM_RAG.spec")
    if spec_file.exists():
        spec_file.unlink()
        print(f"✅ 已清理旧的spec文件")

def build_executable():
    """构建可执行文件"""
    # 应用名称
    app_name = "LLM_RAG"
    # 入口文件
    main_file = "main.py"
    
    # 基本命令
    cmd = [
        "pyinstaller",
        "--name", app_name,
        "--windowed",  # 无控制台窗口
        "--clean",     # 清理PyInstaller缓存
        "--noconfirm"  # 不确认覆盖
    ]
    
    # 添加图标(如果存在)
    icon_path = Path("resources/icon.ico")
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    # 添加数据文件
    # 在Windows上使用分号，在macOS/Linux上使用冒号
    separator = ";" if platform.system() == "Windows" else ":"
    
    # 添加配置文件
    config_path = Path("config")
    if config_path.exists():
        cmd.extend(["--add-data", f"config{separator}config"])
    
    # 添加资源文件
    resources_path = Path("resources")
    if resources_path.exists():
        cmd.extend(["--add-data", f"resources{separator}resources"])
    
    # 添加其他必要的数据文件
    for data_dir in ["data", "schemas", "templates"]:
        data_path = Path(data_dir)
        if data_path.exists():
            cmd.extend(["--add-data", f"{data_dir}{separator}{data_dir}"])
    
    # 添加入口文件
    cmd.append(main_file)
    
    # 执行构建命令
    print(f"🚀 开始构建 {app_name} 可执行文件...")
    try:
        subprocess.check_call(cmd)
        print(f"✅ 构建成功! 可执行文件位置: dist/{app_name}")
        
        # 检查构建结果
        executable_name = f"{app_name}.exe" if platform.system() == "Windows" else app_name
        executable_path = Path(f"dist/{app_name}/{executable_name}")
        if executable_path.exists():
            print(f"✅ 已生成可执行文件: {executable_path}")
            return True
        else:
            print(f"❌ 未找到生成的可执行文件")
            return False
    except Exception as e:
        print(f"❌ 构建失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print(f"🔧 LLM_RAG应用打包工具")
    print("=" * 50)
    
    # 检查PyInstaller
    if not check_pyinstaller():
        return
    
    # 创建并清理构建目录
    create_dist_dir()
    
    # 构建可执行文件
    if build_executable():
        print("\n🎉 打包完成! 应用已准备就绪")
        print("=" * 50)
    else:
        print("\n❌ 打包失败，请检查上述错误信息")
        print("=" * 50)

if __name__ == "__main__":
    main() 