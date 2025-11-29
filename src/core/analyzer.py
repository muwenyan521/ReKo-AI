import random
from collections import Counter, defaultdict
from ..utils.config import get_config


class TextAnalyzer:
    """文本分析器，处理文本统计和词语预测"""
    
    def __init__(self):
        self.corpus = []
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.vocabulary = set()
        self.word_counts = Counter()
        self.is_ready = False
        
        # 强化学习相关参数
        self.reply_history = []
        self.action_counts = defaultdict(Counter)
        self.rewards = defaultdict(float)
        
        # 从配置文件获取强化学习参数
        self.learning_rate = get_config("reinforcement_learning.learning_rate", 0.1)
        self.discount_factor = get_config("reinforcement_learning.discount_factor", 0.9)
        self.epsilon = get_config("reinforcement_learning.exploration_rate", 0.1)
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
    
    def load_corpus(self, texts):
        """加载语料库并构建统计信息"""
        self.corpus = texts
        
        # 重置统计信息
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.vocabulary = set()
        self.word_counts = Counter()
        
        # 处理每个文本，构建n-gram统计
        for text in texts:
            words = text.split()
            if not words:
                continue
            
            self.vocabulary.update(words)
            self.word_counts.update(words)
            
            # 构建二元组和三元组统计
            for i in range(len(words) - 1):
                self.bigram_counts[words[i]][words[i + 1]] += 1
            
            for i in range(len(words) - 2):
                key = (words[i], words[i + 1])
                self.trigram_counts[key][words[i + 2]] += 1
        
        self.is_ready = True
        
        stats = {
            'vocab_size': len(self.vocabulary),
            'total_words': sum(self.word_counts.values()),
            'bigram_pairs': len(self.bigram_counts),
            'trigram_pairs': len(self.trigram_counts)
        }
        return stats
    
    def predict_next(self, context):
        """预测下一个词"""
        if not self.is_ready:
            return []
        
        words = context.split()
        if not words:
            return [word for word, _ in self.word_counts.most_common(5)]
        
        # 优先用三元组预测，其次二元组，最后全局常见词
        if len(words) >= 2:
            trigram_key = (words[-2], words[-1])
            if trigram_key in self.trigram_counts and self.trigram_counts[trigram_key]:
                return [word for word, _ in self.trigram_counts[trigram_key].most_common(5)]
        
        if len(words) >= 1:
            last_word = words[-1]
            if last_word in self.bigram_counts and self.bigram_counts[last_word]:
                return [word for word, _ in self.bigram_counts[last_word].most_common(5)]
        
        return [word for word, _ in self.word_counts.most_common(5)]
    
    def generate_reply(self, query, max_len=20):
        """生成回复文本"""
        if not self.is_ready:
            return "抱歉，我还没有准备好。请先加载技术文档。"
        
        # 从查询开始构建回复
        words = query.split()
        reply = words.copy()
        
        # 记录对话状态
        dialog_states = []
        
        # 生成回复直到达到最大长度
        while len(reply) < max_len:
            # 获取当前上下文
            if len(reply) >= 2:
                context = " ".join(reply[-2:])
            else:
                context = " ".join(reply)
            
            # 预测下一个词
            candidates = self.predict_next(context)
            if not candidates:
                break
            
            # 选择下一个词
            next_word, prob = self.select_action(context, candidates)
            
            # 记录状态
            dialog_states.append((context, next_word, prob))
            self.action_counts[context][next_word] += 1
            
            # 避免重复词
            if next_word not in reply[-3:] and next_word not in [",", "。", "！", "？"] * 2:
                reply.append(next_word)
            else:
                # 尝试其他候选词
                if len(candidates) > 1:
                    other_candidates = [w for w in candidates if w != next_word]
                    if other_candidates:
                        next_word, prob = self.select_action(context, other_candidates)
                        dialog_states.append((context, next_word, prob))
                        self.action_counts[context][next_word] += 1
                        reply.append(next_word)
                    else:
                        break
                else:
                    break
            
            # 遇到结束符号就停止
            if next_word in ["。", "！", "？"]:
                break
        
        # 确保有结束符号
        if reply and reply[-1] not in ["。", "！", "？"]:
            reply.append("。")
        
        # 保存对话历史
        final_reply = " ".join(reply)
        self.reply_history.append((query, final_reply, dialog_states))
        
        # 限制历史记录长度
        if len(self.reply_history) > 1000:
            self.reply_history = self.reply_history[-1000:]
        
        return final_reply
    
    def select_action(self, state, actions):
        """选择动作 - 强化学习策略"""
        # 探索：随机选一个
        if random.random() < self.epsilon:
            action = random.choice(actions)
            prob = 1.0 / len(actions)
            return action, prob
        
        # 利用：基于Q值选择
        q_values = []
        
        # 计算每个动作的Q值
        for act in actions:
            # 历史奖励
            reward = self.rewards.get((state, act), 0)
            
            # 基础值：基于统计频率
            base_val = 0
            words = state.split()
            
            # 优先用三元组
            if len(words) >= 2:
                key = tuple(words[-2:])
                if key in self.trigram_counts:
                    total = sum(self.trigram_counts[key].values())
                    if total > 0:
                        base_val = self.trigram_counts[key].get(act, 0) / total
            # 其次用二元组
            elif words:
                last_word = words[-1]
                if last_word in self.bigram_counts:
                    total = sum(self.bigram_counts[last_word].values())
                    if total > 0:
                        base_val = self.bigram_counts[last_word].get(act, 0) / total
            
            # 多样性奖励：避免总选同一个
            count = self.action_counts[state].get(act, 0) + 1
            diversity = 0.1 / count
            
            # 综合Q值
            q_val = base_val * (1 + reward) + diversity
            q_values.append((q_val, act))
        
        # 按Q值排序
        q_values.sort(reverse=True, key=lambda x: x[0])
        
        # 计算权重
        total = sum(q[0] for q in q_values)
        if total == 0:
            # 如果都是0，就平均分配
            weights = [1.0 / len(q_values)] * len(q_values)
        else:
            weights = [q[0] / total for q in q_values]
        
        # 选择动作
        action_list = [q[1] for q in q_values]
        chosen = random.choices(action_list, weights=weights, k=1)[0]
        
        # 找到对应的概率
        idx = action_list.index(chosen)
        prob = weights[idx]
        
        # 衰减探索率
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return chosen, prob
    
    def update_reward(self, query, reply, reward):
        """更新奖励值 - 强化学习反馈"""
        # 查找对应的对话记录
        for i, (q, r, dialog) in enumerate(self.reply_history):
            if q == query and r == reply:
                # 从后往前更新奖励
                current_reward = reward
                
                # 动态调整学习率
                lr = self.learning_rate
                if abs(reward) > 0.8:  # 强反馈，加大学习率
                    lr *= 1.5
                elif abs(reward) < 0.3:  # 弱反馈，减小学习率
                    lr *= 0.7
                
                # 反向遍历对话状态
                for state, action, prob in reversed(dialog):
                    key = (state, action)
                    old_val = self.rewards.get(key, 0)
                    
                    # 概率越低，更新幅度越大
                    prob_factor = 1.0 / max(prob, 0.1)
                    update = lr * current_reward * prob_factor
                    
                    # 更新奖励值
                    self.rewards[key] = old_val + update
                    
                    # 应用折扣
                    current_reward *= self.discount_factor
                
                # 限制奖励值范围
                for k in list(self.rewards.keys()):
                    self.rewards[k] = max(-1.5, min(1.5, self.rewards[k]))
                
                # 清理旧记录
                if len(self.rewards) > 5000:
                    # 保留最近3000条记录
                    keep_keys = list(self.rewards.keys())[-3000:]
                    for k in list(self.rewards.keys()):
                        if k not in keep_keys:
                            del self.rewards[k]
                
                break
