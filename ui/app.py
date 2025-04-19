import sys
import os

# 将项目根目录添加到Python PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.main_window import main

if __name__ == "__main__":
    main() 