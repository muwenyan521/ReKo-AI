@echo off
chcp 65001
REM ReKo AI 启动脚本 (Windows)
REM 基于神经网络的对话程序

echo ========================================
echo        ReKo AI 启动脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请确保Python已安装并添加到PATH
    echo 请从 https://www.python.org/downloads/ 下载并安装Python 3.8+
    pause
    exit /b 1
)

REM 显示Python版本
echo 检测到Python版本:
python --version

REM 检查是否已安装依赖
echo.
echo 检查依赖包...
python -c "import matplotlib, numpy, jieba, yaml, chardet" >nul 2>&1
if %errorlevel% neq 0 (
    echo 依赖包未安装，正在安装...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
    echo 依赖包安装成功
) else (
    echo 依赖包已安装
)

REM 启动应用程序
echo.
echo 启动 ReKo AI 应用程序...
echo ========================================
python -m src.main

REM 如果应用程序退出，暂停以便查看输出
echo.
echo 应用程序已退出
pause
