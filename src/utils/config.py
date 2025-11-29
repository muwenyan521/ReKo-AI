import os
import logging
from typing import Dict, Any, Optional

from .file_utils import FileUtils

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理类"""
    
    def __init__(self, config_dir: str = "config", config_file: str = "config.yaml"):
        """初始化配置管理器"""
        # 使用绝对路径，确保从项目根目录查找配置文件
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_dir = os.path.join(project_root, config_dir)
        self.config_file = config_file
        self.config_path = os.path.join(self.config_dir, config_file)
        self._config: Dict[str, Any] = {}
        
        os.makedirs(self.config_dir, exist_ok=True)
        self._default_config = self._get_default_config()
        self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "app": {
                "name": "ReKo AI",
                "version": "0.1.2",
                "description": "基于神经网络的对话程序",
                "debug": False
            },
            "gui": {
                "window_width": 1200,
                "window_height": 800,
                "theme": "default",
                "font_family": "Microsoft YaHei",
                "font_size": 10
            },
            "analysis": {
                "max_vocabulary_size": 10000,
                "min_word_frequency": 2,
                "ngram_order": 3,
                "smoothing_alpha": 0.1
            },
            "reinforcement_learning": {
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "exploration_rate": 0.1,
                "max_episodes": 1000
            },
            "paths": {
                "sample_docs": "resources/sample_docs",
                "models": "models",
                "logs": "logs"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "reko_ai.log"
            }
        }
    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                logger.info(f"配置文件不存在，创建默认配置: {self.config_path}")
                success = self.save_config(self._default_config)
                if success:
                    self._config = self._default_config.copy()
                return success
            
            loaded_config = FileUtils.load_yaml_file(self.config_path)
            if loaded_config is None:
                logger.error(f"配置文件加载失败: {self.config_path}")
                return False
            
            # 合并配置：默认配置为基础，加载的配置覆盖
            self._config = self._deep_merge(self._default_config, loaded_config)
            logger.info(f"加载配置文件成功: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            self._config = self._default_config.copy()
            return False
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """保存配置文件"""
        try:
            if config is None:
                config = self._config
            
            success = FileUtils.save_yaml_file(self.config_path, config)
            if success:
                logger.info(f"配置文件已保存: {self.config_path}")
            return success
            
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点分隔符，如 app.name）"""
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        except Exception as e:
            logger.warning(f"获取配置失败 {key}: {e}")
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """设置配置值（支持点分隔符）"""
        try:
            keys = key.split('.')
            current = self._config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            logger.debug(f"配置已更新: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"设置配置失败 {key}: {e}")
            return False
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """批量更新配置"""
        try:
            success_count = 0
            for key, value in updates.items():
                if self.set(key, value):
                    success_count += 1
            
            if success_count == len(updates):
                logger.info(f"批量更新配置完成: {len(updates)} 项")
                return True
            else:
                logger.warning(f"批量更新配置部分失败: {success_count}/{len(updates)}")
                return False
            
        except Exception as e:
            logger.error(f"批量更新配置失败: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()
    
    def reset_to_defaults(self) -> bool:
        """重置为默认配置"""
        try:
            self._config = self._default_config.copy()
            return self.save_config()
        except Exception as e:
            logger.error(f"重置配置失败: {e}")
            return False
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典"""
        result = base.copy()
        
        for key, value in update.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_directories(self) -> bool:
        """创建配置中指定的所有目录"""
        try:
            paths_config = self.get("paths", {})
            success = True
            
            for key, path in paths_config.items():
                if isinstance(path, str):
                    try:
                        os.makedirs(path, exist_ok=True)
                        logger.debug(f"目录已创建: {path}")
                    except Exception as e:
                        logger.error(f"创建目录失败 {path}: {e}")
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"创建目录失败: {e}")
            return False


_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """获取配置值（快捷方式）"""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> bool:
    """设置配置值（快捷方式）"""
    return get_config_manager().set(key, value)
