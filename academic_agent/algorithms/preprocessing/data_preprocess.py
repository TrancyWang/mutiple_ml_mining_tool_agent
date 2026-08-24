"""
这个 Data_PreProcessor 类的主要功能是对文本数据进行预处理，以便后续用于聚类分析（如 KMeans 聚类）。
它提供了两种不同的预处理方式，并支持基于Sentence-BERT/TF-IDF
 的文本向量化方法
 
"""
import jieba
from academic_agent.algorithms.preprocessing.data_clean_rule_enginee import ContentFilter
from typing import List, Optional
import logging
import os
from academic_agent.infrastructure.runtime_paths import bundled_resources_root

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Data_PreProcessor:
    """
    文本预处理工具，支持基于 Sentence-BERT + KMeans 和
    """
    def __init__(self, stopwords_file: Optional[str] = None, user_dict_file: Optional[str] = None):
        # 如果没有提供停用词文件路径，则使用默认路径
        if stopwords_file is None:
            stopwords_file = str(bundled_resources_root() / "data" / "stop_words.txt")
        
        self.stopwords = self.load_stopwords(stopwords_file)
        self.remark_content_filter = ContentFilter()
        jieba.initialize()  # 初始化 jieba
        
        # 如果没有提供用户词典文件路径，则使用默认路径
        if user_dict_file is None:
            user_dict_file = str(bundled_resources_root() / "data" / "dicts.txt")
        
        jieba.load_userdict(user_dict_file)

    def load_stopwords(self, stopwords_path: Optional[str] = None) -> List[str]:
        """加载停用词"""
        if stopwords_path and isinstance(stopwords_path, str):
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                logger.warning(f"停用词文件 {stopwords_path} 未找到，将使用空停用词表。")
                return []
        return []

    def preprocess_data(self, corpus: List[str]) -> List[str]:
        """对文本列表进行分词并过滤停用词"""
        processed_corpus = []
        for line in corpus:
            words = [word for word in jieba.lcut(line.strip()) if word not in self.stopwords]
            processed_corpus.append(' '.join(words))
        return processed_corpus

    def preprocess_data_for_embedding(self, corpus_path: str) -> List[str]:
        """从文件加载原始文本（可扩展为嵌入计算）"""
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.warning(f"文件 {corpus_path} 未找到！")
            return []

    def custom_tokenizer(self, text: str) -> List[str]:
        """自定义分词器（过滤停用词和单字词）"""
        text = self.remark_content_filter.general_filter_content(text, 1, 1000)
        words = jieba.lcut(text)
        return [word for word in words if word not in self.stopwords and len(word) > 1]
