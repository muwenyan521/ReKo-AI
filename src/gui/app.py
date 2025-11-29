
import sys
import threading
import os
import time
import random

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, filedialog
except ImportError as e:
    print(f"无法导入tkinter: {e}")
    print("请确保已安装Python的tkinter库")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.use('TkAgg')
    
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  
except ImportError as e:
    print(f"导入依赖库时出错: {e}")
    print("请确保已安装所有必需的依赖库")
    sys.exit(1)

from ..core.analyzer import TextAnalyzer
from ..utils.config import get_config


class AIDialogApp:
    """AI对话程序主应用 - 基于文本匹配的智能对话系统"""
    
    def __init__(self, root):
        self.root = root
        
        # 从配置获取应用信息
        app_name = get_config("app.name", "ReKo AI")
        app_description = get_config("app.description", "基于文本匹配的智能对话系统")
        self.root.title(f"{app_name} - {app_description}")
        
        # 从配置获取窗口大小
        window_width = get_config("gui.window_width", 1200)
        window_height = get_config("gui.window_height", 800)
        self.root.geometry(f"{window_width}x{window_height}")
        
        # 初始化变量
        self.text_analyzer = TextAnalyzer()
        self.documents = []
        self.is_processing = False
        
        # 创建GUI
        self.create_widgets()
        
        # 初始化可视化图形
        self.init_visualization()
        
        # 设置窗口关闭协议
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """创建GUI组件 - 构建对话界面和控制面板"""
        # 从配置获取字体设置
        font_family = get_config("gui.font_family", "Microsoft YaHei")
        font_size = get_config("gui.font_size", 10)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure("Large.TButton", font=(font_family, font_size + 2))
        
        # 主框架布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 顶部标题
        title_label = ttk.Label(main_frame, text="ReKo AI", font=(font_family, font_size + 14, "bold"))
        title_label.pack(pady=(0, 15))
        
        # 水平分割框架
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧对话区域
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 右侧控制面板
        right_frame = ttk.Frame(content_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)  # 固定宽度
        
        # 对话显示区域
        dialog_label = ttk.Label(left_frame, text="对话窗口", font=(font_family, font_size + 6, "bold"))
        dialog_label.pack(anchor=tk.W)
        
        self.dialog_display = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, height=25, font=(font_family, font_size + 2))
        self.dialog_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.dialog_display.config(state=tk.DISABLED)
        
        # 用户输入区域
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.user_input = ttk.Entry(input_frame, font=(font_family, font_size + 2))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=8)
        self.user_input.bind("<Return>", self.send_message)
        
        send_button = ttk.Button(input_frame, text="发送", command=self.send_message, style="Large.TButton")
        send_button.pack(side=tk.RIGHT, ipady=5)
        
        # 右侧控制面板内容
        control_label = ttk.Label(right_frame, text="控制面板", font=(font_family, font_size + 6, "bold"))
        control_label.pack(pady=(0, 15))
        
        # 文档加载按钮
        load_docs_button = ttk.Button(right_frame, text="加载技术文档", command=self.load_documents, style="Large.TButton")
        load_docs_button.pack(fill=tk.X, pady=(0, 10), ipady=6)
        
        # 预处理按钮
        self.process_button = ttk.Button(right_frame, text="处理文档", command=self.process_documents, style="Large.TButton")
        self.process_button.pack(fill=tk.X, pady=(0, 10), ipady=6)
        
        # 预测按钮
        predict_button = ttk.Button(right_frame, text="预测下一个词", command=self.predict_next_word, style="Large.TButton")
        predict_button.pack(fill=tk.X, pady=(0, 15), ipady=6)
        
        # 状态显示
        status_label = ttk.Label(right_frame, text="状态:", font=(font_family, font_size + 4, "bold"))
        status_label.pack(anchor=tk.W, pady=(15, 5))
        
        self.status_display = ttk.Label(right_frame, text="未初始化", relief=tk.SUNKEN, font=(font_family, font_size + 2))
        self.status_display.pack(fill=tk.X, pady=(0, 15), ipady=5)
        
        # 统计信息区域
        stats_label = ttk.Label(right_frame, text="文档统计信息", font=(font_family, font_size + 6, "bold"))
        stats_label.pack(pady=(15, 5))
        
        self.stats_frame = ttk.Frame(right_frame)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 统计标签
        self.vocab_label = ttk.Label(self.stats_frame, text="词汇量: -", anchor=tk.W, font=(font_family, font_size + 2))
        self.vocab_label.pack(fill=tk.X, pady=4)
        
        self.word_label = ttk.Label(self.stats_frame, text="总词数: -", anchor=tk.W, font=(font_family, font_size + 2))
        self.word_label.pack(fill=tk.X, pady=4)
        
        self.bigram_label = ttk.Label(self.stats_frame, text="二元组数量: -", anchor=tk.W, font=(font_family, font_size + 2))
        self.bigram_label.pack(fill=tk.X, pady=4)
        
        self.trigram_label = ttk.Label(self.stats_frame, text="三元组数量: -", anchor=tk.W, font=(font_family, font_size + 2))
        self.trigram_label.pack(fill=tk.X, pady=4)
        
        # 可视化区域标题
        vis_label = ttk.Label(right_frame, text="神经网络推理进度", font=(font_family, font_size + 6, "bold"))
        vis_label.pack(pady=(15, 5))
        
        # 可视化图形区域
        self.fig_frame = ttk.Frame(right_frame)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def update_stats_display(self, stats):
        """更新统计信息显示"""
        self.vocab_label.config(text=f"词汇量: {stats.get('vocab_size', 0)}")
        self.word_label.config(text=f"总词数: {stats.get('total_words', 0)}")
        self.bigram_label.config(text=f"二元组数量: {stats.get('bigram_pairs', 0)}")
        self.trigram_label.config(text=f"三元组数量: {stats.get('trigram_pairs', 0)}")
    
    def init_visualization(self):
        """初始化可视化图形 - 创建matplotlib图表并嵌入到Tkinter界面"""
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.set_title("推理进度", fontsize=14)
        self.ax.set_xlabel("时间", fontsize=12)
        self.ax.set_ylabel("匹配度", fontsize=12)
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        
        # 嵌入到Tkinter中
        self.canvas = FigureCanvasTkAgg(self.fig, self.fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化数据
        self.time_points = []
        self.match_scores = []
        self.current_time = 0
    
    def update_visualization(self, score):
        """更新可视化图形 - 添加新数据点并刷新图表显示"""
        self.current_time += 1
        self.time_points.append(self.current_time)
        self.match_scores.append(score)
        
        # 保持数据点数量在合理范围内
        if len(self.time_points) > 50:
            self.time_points = self.time_points[-50:]
            self.match_scores = self.match_scores[-50:]
        
        # 更新图表
        self.ax.clear()
        self.ax.plot(self.time_points, self.match_scores, 'b-')
        self.ax.set_title("推理进度")
        self.ax.set_xlabel("时间")
        self.ax.set_ylabel("匹配度")
        
        # 设置y轴范围以保持图表美观
        if self.match_scores:
            min_score = min(self.match_scores)
            max_score = max(self.match_scores)
            padding = (max_score - min_score) * 0.1 if max_score > min_score else 0.1
            self.ax.set_ylim(max(0, min_score - padding), max_score + padding)
        
        self.canvas.draw()
        
    def load_documents(self):
        """加载技术文档 - 选择文件夹并异步加载所有TXT文件"""
        folder_path = filedialog.askdirectory(title="选择包含TXT文档的技术文档文件夹")
        if folder_path:
            self.add_message("系统", f"正在加载文档文件夹: {folder_path}")
            self.status_display.config(text="正在加载文档...")
            
            # 在线程中加载文档以避免阻塞UI
            threading.Thread(target=self._load_documents_thread, args=(folder_path,), daemon=True).start()
    
    def _load_documents_thread(self, folder_path):
        """在后台线程中加载文档 - 遍历文件夹读取所有TXT文件内容"""
        try:
            documents = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():
                                    documents.append(content)
                        except Exception as e:
                            print(f"读取文件 {file_path} 出错: {e}")
            
            self.documents = documents
            
            # 更新UI
            self.root.after(0, lambda: self.add_message("系统", f"成功加载 {len(documents)} 个文档"))
            self.root.after(0, lambda: self.status_display.config(text=f"已加载 {len(documents)} 个文档"))
            
        except Exception as e:
            self.root.after(0, lambda: self.add_message("系统", f"加载文档出错: {str(e)}"))
            self.root.after(0, lambda: self.status_display.config(text="加载失败"))
    
    def process_documents(self):
        """处理文档并构建统计信息 - 异步调用文本分析器处理文档"""
        if not self.documents:
            self.add_message("系统", "请先加载技术文档!")
            return
            
        if self.is_processing:
            self.add_message("系统", "正在处理中...")
            return
            
        self.is_processing = True
        self.process_button.config(state=tk.DISABLED, text="处理中...")
        self.add_message("系统", "开始处理文档...")
        self.status_display.config(text="正在处理文档...")
        
        # 在线程中处理文档
        threading.Thread(target=self._process_documents_thread, daemon=True).start()
    
    def _process_documents_thread(self):
        """在后台线程中处理文档 - 调用文本分析器构建统计信息"""
        try:
            # 处理文档
            stats = self.text_analyzer.load_corpus(self.documents)
            
            # 更新UI
            self.root.after(0, lambda s=stats: self.update_stats_display(s))
            self.root.after(0, lambda: self.add_message("系统", f"文档处理完成! 词汇量: {stats['vocab_size']}"))
            self.root.after(0, lambda: self.status_display.config(text="文档处理完成"))
            
        except Exception as e:
            self.root.after(0, lambda: self.add_message("系统", f"处理文档出错: {str(e)}"))
            self.root.after(0, lambda: self.status_display.config(text="处理失败"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL, text="处理文档"))
    
    def predict_next_word(self):
        """预测下一个词 - 基于文本分析器预测用户输入的下一个可能词语"""
        if not self.text_analyzer.is_ready:
            self.add_message("系统", "请先加载并处理技术文档!")
            return
            
        # 获取用户输入
        user_text = self.user_input.get()
        if not user_text.strip():
            self.add_message("系统", "请输入一些文本以进行预测!")
            return
            
        try:
            # 预测下一个词
            next_words = self.text_analyzer.predict_next(user_text)
            
            if not next_words:
                self.add_message("ReKo AI", "没有找到匹配的预测词。")
            else:
                # 显示结果
                result_text = "可能的下一个词:\n"
                for i, word in enumerate(next_words[:5]):
                    # 由于使用统计方法，我们无法计算精确概率，可以使用相对频率
                    result_text += f"{i+1}. {word}\n"
                
                self.add_message("ReKo AI", result_text)
                
        except Exception as e:
            self.add_message("系统", f"预测出错: {str(e)}")
    
    def send_message(self, event=None):
        """发送消息 - 处理用户输入并触发AI回复生成"""
        user_text = self.user_input.get()
        if user_text.strip():
            self.add_message("用户", user_text)
            self.user_input.delete(0, tk.END)
            
            # 如果文档已处理，自动生成回复
            if self.text_analyzer.is_ready:
                # 在生成回复时更新可视化
                try:
                    # 模拟匹配度分数更新 - 模拟神经网络推理过程
                    for i in range(10):  # 模拟10个处理步骤
                        # 计算一个模拟的匹配度分数，先上升后趋于稳定
                        if i < 3:
                            score = 0.2 + (i * 0.15) + random.uniform(-0.03, 0.03)
                        elif i < 7:
                            score = 0.65 + (i-3) * 0.05 + random.uniform(-0.02, 0.02)
                        else:
                            score = 0.85 + random.uniform(-0.01, 0.01)
                        self.update_visualization(score)
                        self.root.update()
                        time.sleep(0.1)  # 短暂延迟展示动画效果
                except:
                    pass  # 即使可视化更新失败也继续生成回复
                
                self.generate_response(user_text)
    
    def generate_response(self, user_text):
        """生成回复 - 调用文本分析器生成AI回复"""
        try:
            # 使用TextAnalyzer生成回复
            response = self.text_analyzer.generate_reply(user_text)
            
            self.add_message("ReKo AI", response)
                
        except Exception as e:
            self.add_message("ReKo AI", f"生成回复时出错: {str(e)}")
    
    def add_message(self, sender, message):
        """添加消息到对话窗口 - 显示消息并为AI回复添加评分按钮"""
        self.dialog_display.config(state=tk.NORMAL)
        
        # 插入发送者和消息
        self.dialog_display.insert(tk.END, f"[{sender}]: {message}\n")
        
        # 如果是AI的回复，添加评分按钮
        if sender == "ReKo AI":
            # 保存当前的消息位置
            message_start = self.dialog_display.index(tk.END)
            
            # 创建评分按钮的框架
            button_frame = tk.Frame(self.dialog_display, bg="#f0f0f0")
            button_frame.pack_propagate(True)
            
            # 从配置获取字体设置
            font_family = get_config("gui.font_family", "Microsoft YaHei")
            font_size = get_config("gui.font_size", 10)
            
            # 添加点赞按钮
            like_button = tk.Button(button_frame, text="👍 有用", 
                                  command=lambda msg=message: self.rate_reply(msg, 1.0),
                                  bg="#4CAF50", fg="white", width=10, height=1, font=(font_family, font_size + 1))
            like_button.pack(side=tk.LEFT, padx=10, pady=5)
            
            # 添加点踩按钮
            dislike_button = tk.Button(button_frame, text="👎 没用", 
                                     command=lambda msg=message: self.rate_reply(msg, -0.5),
                                     bg="#F44336", fg="white", width=10, height=1, font=(font_family, font_size + 1))
            dislike_button.pack(side=tk.LEFT, padx=10, pady=5)
            
            # 将按钮框架嵌入到文本框中
            self.dialog_display.window_create(tk.END, window=button_frame)
            
            # 保存消息和对应的评分按钮信息
            if not hasattr(self, 'messages_with_ratings'):
                self.messages_with_ratings = []
            self.messages_with_ratings.append((message, button_frame, like_button, dislike_button))
        
        self.dialog_display.insert(tk.END, "\n")
        self.dialog_display.config(state=tk.DISABLED)
        self.dialog_display.see(tk.END)
    
    def rate_reply(self, message, rating):
        """处理用户对回复的评分 - 更新强化学习奖励并禁用评分按钮"""
        # 查找对应的查询（用户的最后一条消息）
        if len(self.messages_with_ratings) > 0:
            # 调用text_analyzer的update_reward方法更新奖励
            # 获取最后一次对话的用户输入（这里简化处理，实际应该更精确地匹配对话对）
            user_query = ""  # 这里需要完善，实际应该跟踪完整的对话历史
            
            # 如果有最近的对话历史
            if hasattr(self.text_analyzer, 'reply_history') and self.text_analyzer.reply_history:
                # 查找最匹配的对话
                for hist_query, hist_reply, _ in reversed(self.text_analyzer.reply_history):
                    if message in hist_reply or hist_reply in message:  # 简化的匹配逻辑
                        user_query = hist_query
                        break
            
            # 更新奖励
            if user_query:
                self.text_analyzer.update_reward(user_query, message, rating)
                
                # 反馈给用户
                feedback = "感谢反馈！我会努力改进的。" if rating > 0 else "抱歉，我会继续学习改进。"
                self.add_message("系统", feedback)
            
            # 禁用已评分的按钮，防止重复评分
            for msg, frame, like_btn, dislike_btn in self.messages_with_ratings:
                if msg == message:
                    like_btn.config(state=tk.DISABLED, bg="#CCCCCC")
                    dislike_btn.config(state=tk.DISABLED, bg="#CCCCCC")
                    break
    
    def on_closing(self):
        """窗口关闭时的处理函数 - 清理资源并退出程序"""
        # 清理资源
        plt.close('all')  # 关闭所有matplotlib图形
        # 销毁窗口并退出程序
        self.root.destroy()
        import sys
        sys.exit(0)  # 强制退出Python进程
