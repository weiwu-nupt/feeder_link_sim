"""
馈电链路仿真平台 - 主程序入口
Feeder Link Simulation Platform - Main Entry Point
"""

import sys
import os

# 确保项目根目录在 sys.path 中，无论从哪里启动都能找到 modules/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("馈电链路仿真平台")
    app.setApplicationVersion("0.1.0")

    # 设置全局默认字体，避免 QFont point size <= 0 警告
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()