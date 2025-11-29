#!/usr/bin/env python3
"""
ReKo AI 安装脚本
基于神经网络的对话程序
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# 读取requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="reko-ai",
    version="0.1.2",
    description="基于神经网络的对话程序",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MemoTeam",
    url="https://github.com/Janmast-eng/ReKo-AI",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
    ],
    keywords=[
        "nlp", "ai", "chatbot", "reinforcement-learning", 
        "text-analysis", "chinese-nlp", "neural-network"
    ],
    entry_points={
        'console_scripts': [
            'reko-ai=src.main:main',
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Janmast-eng/ReKo-AI/issues",
        "Source": "https://github.com/Janmast-eng/ReKo-AI",
    },
)
