#!/usr/bin/env python3
"""ReKo AI - 基于神经网络的对话程序主入口"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gui.app import AIDialogApp
from src.utils.config import get_config_manager


def main():
    """启动ReKo AI应用程序"""
    try:
        # 初始化配置管理器
        config_manager = get_config_manager()
        
        # 从配置获取日志设置
        log_level = config_manager.get("logging.level", "INFO")
        log_format = config_manager.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file_name = config_manager.get("logging.file", "reko_ai.log")
        logs_dir = config_manager.get("paths.logs", "logs")
        
        # 确保logs目录存在
        os.makedirs(logs_dir, exist_ok=True)
        
        # 构建完整的日志文件路径
        log_file_path = os.path.join(logs_dir, log_file_name)
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info("启动 ReKo AI 应用程序")
        
        # 从配置获取应用信息
        app_name = config_manager.get("app.name", "ReKo AI")
        app_version = config_manager.get("app.version", "0.1.2")
        app_description = config_manager.get("app.description", "基于神经网络的对话程序")
        
        root = tk.Tk()
        root.title(f"{app_name} - {app_description}")
        
        # 从配置获取窗口大小
        window_width = config_manager.get("gui.window_width", 1200)
        window_height = config_manager.get("gui.window_height", 800)
        root.geometry(f"{window_width}x{window_height}")
        
        app = AIDialogApp(root)
        logger.info("GUI主循环启动")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}")
        messagebox.showerror("错误", f"应用程序启动失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
