import re
import logging
import jieba
from collections import Counter

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        logger.info("预处理器初始化完成")
    
    def _load_stop_words(self):
        """加载停用词表"""
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们', '你们', '他们', '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '因为', '所以', '但是', '然后', '如果', '虽然', '可以', '应该', '能够', '已经', '正在', '将要', '可能', '一定', '必须', '需要', '想要', '希望', '觉得', '认为', '知道', '了解', '理解', '记得', '忘记', '开始', '结束', '完成', '进行', '继续', '停止', '改变', '增加', '减少', '提高', '降低', '改善', '恶化', '成功', '失败', '正确', '错误', '重要', '必要', '主要', '次要', '基本', '高级', '简单', '复杂', '容易', '困难', '快速', '缓慢', '大', '小', '多', '少', '长', '短', '高', '低', '强', '弱', '新', '旧', '好', '坏', '对', '错', '真', '假', '是', '否', '有', '无', '存在', '消失', '出现', '发生', '变成', '成为', '作为', '对于', '关于', '根据', '按照', '通过', '使用', '利用', '应用', '实施', '执行', '操作', '处理', '管理', '控制', '监督', '检查', '测试', '评估', '分析', '研究', '开发', '设计', '制造', '生产', '销售', '购买', '消费', '服务', '支持', '帮助', '指导', '教育', '培训', '学习', '工作', '生活', '健康', '安全', '环境', '经济', '社会', '文化', '政治', '科技', '艺术', '历史', '未来', '现在', '过去', '时间', '空间', '物质', '精神', '思想', '情感', '行为', '结果', '效果', '影响', '作用', '意义', '价值', '目标', '目的', '方法', '手段', '工具', '资源', '信息', '知识', '技能', '能力', '经验', '水平', '质量', '数量', '速度', '效率', '效益', '成本', '价格', '价值', '利润', '收入', '支出', '投资', '回报', '风险', '机会', '挑战', '问题', '解决方案', '策略', '计划', '项目', '任务', '活动', '事件', '情况', '状态', '条件', '因素', '原因', '结果', '过程', '阶段', '步骤', '环节', '部分', '整体', '系统', '结构', '功能', '性能', '特性', '特点', '优势', '劣势', '机会', '威胁', '内部', '外部', '宏观', '微观', '全局', '局部', '长期', '短期', '直接', '间接', '主动', '被动', '积极', '消极', '正面', '负面', '有利', '不利', '相同', '不同', '相似', '相反', '相关', '无关', '重要', '不重要', '必要', '不必要', '可能', '不可能', '可以', '不可以', '应该', '不应该', '必须', '不必', '需要', '不需要', '想要', '不想要', '希望', '不希望', '觉得', '不觉得', '认为', '不认为', '知道', '不知道', '了解', '不了解', '理解', '不理解', '记得', '不记得', '忘记', '不忘记'
        }
        return stop_words
    
    def clean_text(self, text):
        """清洗文本 - 移除HTML、URL、特殊字符等"""
        try:
            # 移除HTML标签、URL、邮箱和特殊字符，保留中文、英文、数字和基本标点
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：""''（）《》【】、]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            logger.debug("文本清洗完成")
            return text
            
        except Exception as e:
            logger.error(f"文本清洗失败: {e}")
            return text
    
    def segment_text(self, text, use_jieba=True):
        """中文分词"""
        try:
            if not text.strip():
                return []
            
            # 使用jieba分词或简单空格分词
            words = jieba.lcut(text) if use_jieba else text.split()
            
            logger.debug(f"分词完成，共 {len(words)} 个词")
            return words
            
        except Exception as e:
            logger.error(f"分词失败: {e}")
            return text.split()
    
    def remove_stop_words(self, words):
        """移除停用词"""
        try:
            # 过滤停用词和单字词
            result = []
            for w in words:
                if w not in self.stop_words and len(w) > 1:
                    result.append(w)
            
            removed = len(words) - len(result)
            logger.debug(f"移除了 {removed} 个停用词")
            return result
            
        except Exception as e:
            logger.error(f"移除停用词出错: {e}")
            return words
    
    def normalize_text(self, text, to_lower=True):
        """文本标准化处理"""
        try:
            if to_lower:
                text = text.lower()
            
            # 标点符号标准化：逗号、句号、感叹号、问号
            text = re.sub(r'[，,]+', '，', text)
            text = re.sub(r'[。.]+', '。', text)
            text = re.sub(r'[！!]+', '！', text)
            text = re.sub(r'[？?]+', '？', text)
            
            logger.debug("文本标准化完成")
            return text
            
        except Exception as e:
            logger.error(f"标准化出错: {e}")
            return text
    
    def preprocess_pipeline(self, text, steps=None):
        """文本预处理流水线"""
        if steps is None:
            steps = ['clean', 'normalize', 'segment', 'remove_stop_words']
        
        try:
            temp_text = text
            words = []
            
            # 按步骤处理
            for step in steps:
                if step == 'clean':
                    temp_text = self.clean_text(temp_text)
                elif step == 'normalize':
                    temp_text = self.normalize_text(temp_text)
                elif step == 'segment':
                    words = self.segment_text(temp_text)
                elif step == 'remove_stop_words':
                    if words:  # 确保已经有分词结果
                        words = self.remove_stop_words(words)
            
            # 如果没分词，最后分一下
            if not words:
                words = self.segment_text(temp_text)
            
            logger.info(f"预处理完成: {len(text)} -> {len(words)} 词")
            return words
            
        except Exception as e:
            logger.error(f"预处理出错: {e}")
            return []
    
    def build_vocabulary(self, texts, min_freq=None, max_size=None):
        """构建词汇表"""
        try:
            # 从配置获取参数，如果没有提供则使用配置中的默认值
            from ..utils.config import get_config
            if min_freq is None:
                min_freq = get_config("analysis.min_word_frequency", 2)
            if max_size is None:
                max_size = get_config("analysis.max_vocabulary_size", 10000)
            
            counter = Counter()
            
            # 统计词频并过滤低频词，按词频排序并限制词汇表大小
            for text in texts:
                words = self.preprocess_pipeline(text)
                counter.update(words)
            
            filtered = {}
            for word, count in counter.items():
                if count >= min_freq:
                    filtered[word] = count
            
            sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_items) > max_size:
                sorted_items = sorted_items[:max_size]
            
            vocab = {}
            for idx, (word, _) in enumerate(sorted_items):
                vocab[word] = idx
            
            logger.info(f"词汇表构建完成: {len(vocab)} 个词 (min_freq={min_freq}, max_size={max_size})")
            return vocab
            
        except Exception as e:
            logger.error(f"构建词汇表出错: {e}")
            return {}
    
    def text_to_sequence(self, text, vocab):
        """将文本转换为词索引序列"""
        try:
            words = self.preprocess_pipeline(text)
            seq = []
            for word in words:
                if word in vocab:
                    seq.append(vocab[word])
            
            logger.debug(f"文本转序列: {len(seq)} 个词")
            return seq
            
        except Exception as e:
            logger.error(f"序列化出错: {e}")
            return []
    
    def calculate_text_statistics(self, text):
        """计算文本统计信息"""
        try:
            words = self.preprocess_pipeline(text)
            chars = list(text)
            
            # 计算字符数、词数、句子数、唯一词数、平均词长和词汇丰富度
            char_count = len(chars)
            word_count = len(words)
            sentence_count = len(re.split(r'[。！？.!?]', text))
            unique_words = len(set(words))
            
            avg_len = sum(len(w) for w in words) / len(words) if words else 0
            richness = unique_words / len(words) if words else 0
            
            stats = {
                'char_count': char_count,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'unique_words': unique_words,
                'avg_word_length': avg_len,
                'vocabulary_richness': richness
            }
            
            logger.debug(f"统计计算完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"统计计算出错: {e}")
            return {}


# 全局预处理器实例
_preprocessor = None


def get_preprocessor():
    """获取全局预处理器实例"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor


def preprocess_text(text, steps=None):
    """预处理文本（便捷函数）"""
    if steps is None:
        steps = ['clean', 'normalize', 'segment', 'remove_stop_words']
    return get_preprocessor().preprocess_pipeline(text, steps)
