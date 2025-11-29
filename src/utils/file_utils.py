import os
import logging
import json
import yaml
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class FileUtils:
    """文件工具类"""
    
    @staticmethod
    def read_text_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """读取文本文件"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            logger.info(f"读取文件成功: {file_path}")
            return content
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def write_text_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """写入文本文件"""
        try:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            logger.info(f"写入文件成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"写入文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"加载JSON文件成功: {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载JSON文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def save_json_file(file_path: str, data: Dict[str, Any], indent: int = 2) -> bool:
        """保存JSON文件"""
        try:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            logger.info(f"保存JSON文件成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存JSON文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_yaml_file(file_path: str) -> Optional[Dict[str, Any]]:
        """加载YAML文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            logger.info(f"加载YAML文件成功: {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载YAML文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def save_yaml_file(file_path: str, data: Dict[str, Any]) -> bool:
        """保存YAML文件"""
        try:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"保存YAML文件成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存YAML文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_list(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """获取目录文件列表"""
        try:
            if not os.path.exists(directory):
                logger.warning(f"目录不存在: {directory}")
                return []
            
            file_list = []
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path):
                    if extensions is None:
                        file_list.append(file_path)
                    else:
                        for ext in extensions:
                            if filename.lower().endswith(ext.lower()):
                                file_list.append(file_path)
                                break
            
            logger.info(f"在 {directory} 中找到 {len(file_list)} 个文件")
            return file_list
        except Exception as e:
            logger.error(f"获取文件列表失败 {directory}: {e}")
            return []
