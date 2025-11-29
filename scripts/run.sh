#!/bin/bash
# ReKo AI 启动脚本 (Linux/Unix)
# 基于神经网络的对话程序

echo "========================================"
echo "       ReKo AI 启动脚本"
echo "========================================"
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请确保Python3已安装"
    echo "请使用包管理器安装Python3 (例如: sudo apt install python3)"
    exit 1
fi

# 显示Python版本
echo "检测到Python版本:"
python3 --version

# 检查是否已安装依赖
echo
echo "检查依赖包..."
python3 -c "import matplotlib, numpy, jieba, yaml, chardet" &> /dev/null
if [ $? -ne 0 ]; then
    echo "依赖包未安装，正在安装..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖包安装失败"
        exit 1
    fi
    echo "依赖包安装成功"
else
    echo "依赖包已安装"
fi

# 启动应用程序
echo
echo "启动 ReKo AI 应用程序..."
echo "========================================"
python3 -m src.main
