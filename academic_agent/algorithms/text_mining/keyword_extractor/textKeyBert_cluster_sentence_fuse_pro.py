# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔥 Qt / GUI 稳定性关键（Intel Mac 必加）
os.environ["QT_MAC_WANTS_LAYER"] = "1"
os.environ["QT_QPA_PLATFORM"] = "cocoa"
import torch
import sys
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import logging
import jieba  # 在模块级别导入 jieba

# 添加 src_code 到 Python 路径 - 使用绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(base_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from academic_agent.algorithms.preprocessing.data_clean_rule_enginee import ContentFilter
from academic_agent.infrastructure.model_paths import default_model_root
from academic_agent.infrastructure.runtime_paths import bundled_resources_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextKeyBert(object):

    def __init__(self, dict_file=None, stopwords_file=None):
        # 使用 ContentFilter 进行文本过滤
        self.remark_content_filter = ContentFilter()
        
        # 加载停用词（支持自定义路径）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if stopwords_file is None:
            stopwords_file = str(bundled_resources_root() / "data" / "stop_words.txt")
        self.stopwords = self.load_stopwords(stopwords_file)
        
        # 加载自定义词典（支持自定义路径）
        if dict_file is None:
            dict_file = str(bundled_resources_root() / "data" / "dicts.txt")
        self.load_user_dict(dict_file)
        
        # 设置聚类用的 embedding 模型路径（使用 bge-cn 模型）
        self.embedding_model_path = str(default_model_root() / "bge-cn")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.embedding_model = SentenceTransformer(
            self.embedding_model_path,
            device=self.device
        )

        self.custom_tokenizer = lambda text: self.custom_tokenize(text)

        self.vectorizer = CountVectorizer(
            tokenizer=self.custom_tokenizer,
            token_pattern=None
        )

        self.kbert_model = KeyBERT(model=self.embedding_model)

        # ===== 核心参数 =====
        self.sentence_recall_topn = 20  # 从 15 增加到 20
        self.ngram_range = (2, 5)
        self.diversity = 0.5

        self.alpha = 0.6
        self.minority_alpha = 0.2

        self.beta = 0.95   # sentence vs doc
    
    # ======================
    # utils
    # ======================
    def load_stopwords(self, stopwords_path):
        """加载停用词"""
        try:
            if os.path.exists(stopwords_path):
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.warning(f"加载停用词失败：{e}")
        return []
    
    def load_user_dict(self, dict_path):
        """加载用户自定义词典"""
        try:
            if os.path.exists(dict_path):
                with open(dict_path, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f if line.strip()]
                # 批量加载词典
                for word in words:
                    jieba.add_word(word)
                logger.info(f"✅ 已加载自定义词典：{dict_path}，共 {len(words)} 个词")
            else:
                logger.warning(f"自定义词典文件不存在：{dict_path}")
        except Exception as e:
            logger.error(f"加载自定义词典失败：{e}")
    
    def custom_tokenize(self, text):
        """自定义分词器（使用 jieba 并过滤停用词）"""
        # 先使用 ContentFilter 进行文本清洗
        text = self.remark_content_filter.general_filter_content(text, 1, 1000)
        # 使用 jieba 分词
        words = jieba.lcut(text)
        # 过滤停用词，保留双字及以上
        filtered_words = []
        for word in words:
            # 跳过停用词
            if word in self.stopwords:
                continue
            # 保留双字及以上的词
            if len(word) >= 2:
                filtered_words.append(word)
            # 或者是有意义的单字（不是标点、数字、英文）
            elif len(word) == 1 and word not in '，。！？；：''、（）()[]{}0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                filtered_words.append(word)
        return filtered_words
    
    def normalize_scores(self, score_dict: dict) -> dict:
        if not score_dict:
            return {}

        min_score = min(score_dict.values())
        max_score = max(score_dict.values())

        if min_score == max_score:
            return {k: 1.0 for k in score_dict}

        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in score_dict.items()
        }

    def safe_extract_keywords(self, text, topn, candidates=None):
        if not text or not text.strip():
            return []

        try:
            return self.kbert_model.extract_keywords(
                text,
                vectorizer=None if candidates else self.vectorizer,
                candidates=candidates,
                top_n=topn,
                keyphrase_ngram_range=self.ngram_range,
                use_mmr=True,
                diversity=self.diversity
            )
        except ValueError:
            return []

    # ======================
    # 🔥 短词控制
    # ======================
    def is_valid_short_phrase(self, phrase):
        return True

    # ======================
    # 🔥 候选短语
    # ======================
    def generate_phrase_candidates(self, docs):
        phrases = set()

        for doc in docs:
            tokens = self.custom_tokenizer(doc)

            for i in range(len(tokens)):
                for n in range(2, 5):
                    if i + n <= len(tokens):
                        span = tokens[i:i+n]

                        if any(len(w) == 1 for w in span):
                            continue

                        phrase = "".join(span)

                        if 2 <= len(phrase) <= 14 and self.is_valid_short_phrase(phrase):
                            phrases.add(phrase)

        return list(phrases)[:2000]  # 从 1000 增加到 2000

    # ======================
    # 🔥 句子片段（收紧）
    # ======================
    def generate_sentence_spans(self, docs):
        spans = set()

        for doc in docs:
            doc = doc.strip()
            if len(doc) < 6:
                continue

            parts = doc.replace("。", ",").replace("，", ",").split(",")

            for part in parts:
                part = part.strip()

                if 3 <= len(part) <= 20:  # 从 4-15 改为 3-20
                    spans.add(part)

        return list(spans)[:500]  # 从 300 增加到 500

    # ======================
    # 🔥 sentence-level
    # ======================
    def extract_sentence_level_keywords(self, docs):

        keyword_score_sum = defaultdict(float)
        keyword_doc_set = defaultdict(set)

        merged_text = " ".join(docs)

        candidates = self.generate_phrase_candidates(docs)
        candidates += self.generate_sentence_spans(docs)
        candidates = list(set(candidates))

        keywords = self.safe_extract_keywords(
            merged_text,
            self.sentence_recall_topn * 4,  # 从 3 倍增加到 4 倍
            candidates=candidates
        )

        for kw, score in keywords:
            score = float(score) if score else 0.0

            for doc_id, doc in enumerate(docs):
                if kw in doc:
                    keyword_score_sum[kw] += score
                    keyword_doc_set[kw].add(doc_id)

        total_docs = len(docs)
        fused_score = {}

        for kw in keyword_score_sum:
            coverage = len(keyword_doc_set[kw])

            # 移除覆盖率限制，允许所有关键词
            # if coverage < 1:
            #     continue

            avg_score = keyword_score_sum[kw] / coverage
            coverage_ratio = coverage / total_docs

            base_score = self.alpha * avg_score + (1 - self.alpha) * coverage_ratio
            minority_boost = 1.0 + self.minority_alpha * (1 - coverage_ratio)

            fused_score[kw] = base_score * minority_boost

        return fused_score

    # ======================
    # 🔥 doc-level
    # ======================
    def extract_doc_level_keywords(self, docs):
        """从文档级提取关键词，带异常处理"""
        try:
            merged_text = " ".join(docs)
            logger.info(f"   [DOC-LEVEL] 合并文本长度：{len(merged_text)}")
            
            keywords = self.safe_extract_keywords(
                merged_text,
                self.sentence_recall_topn * 5  # 从 3 倍增加到 5 倍，提高召回
            )
            
            logger.info(f"   [DOC-LEVEL] 提取到 {len(keywords)} 个关键词")
            
            doc_scores = {}
            for kw, score in keywords:
                if len(kw) < 2:  # 从 3 降低到 2，允许双字词
                    continue
                doc_scores[kw] = float(score)
            
            return doc_scores
            
        except ValueError as e:
            error_msg = str(e)
            if "empty vocabulary" in error_msg.lower():
                logger.warning(f"   ⚠️  [DOC-LEVEL] 空词汇表错误，返回空字典")
                return {}
            else:
                # 其他错误重新抛出
                raise

    # ======================
    # 🔥 融合
    # ======================
    def fuse_sentence_doc_scores(self, sent_scores, doc_scores):

        fused = {}
        all_keys = set(sent_scores) | set(doc_scores)

        for kw in all_keys:
            s_score = sent_scores.get(kw, 0)
            d_score = doc_scores.get(kw, 0)

            fused[kw] = self.beta * s_score + (1 - self.beta) * d_score

        return fused

    # ======================
    # 🔥 清洗（移除主题词强化）
    # ======================
    def clean_keywords(self, score_dict):

        cleaned = {}

        for kw, score in score_dict.items():

            if kw.isascii():
                continue

            cleaned[kw] = score

        return cleaned

    # ======================
    # 主函数
    # ======================
    def get_key_words(self, docs, res_top_n=50):  # 从 35 增加到 50

        if not docs or all(not d.strip() for d in docs):
            return []
        
        # 调试信息：检查输入数据
        print(f"\n🔍 [DEBUG] 输入文档数量：{len(docs)}")
        print(f"🔍 [DEBUG] 第一篇文档：{docs[0][:50] if docs else 'N/A'}...")
        
        # Step1: 句子级和文档级关键词提取
        sent_scores = self.extract_sentence_level_keywords(docs)
        doc_scores = self.extract_doc_level_keywords(docs)
        print(f"🔍 [DEBUG] 句子级关键词：{len(sent_scores)} 个，文档级关键词：{len(doc_scores)} 个")
        
        # Step2: 归一化
        sent_scores = self.normalize_scores(sent_scores)
        doc_scores = self.normalize_scores(doc_scores)
        
        # Step3: 融合
        fused_score = self.fuse_sentence_doc_scores(sent_scores, doc_scores)
        print(f"🔍 [DEBUG] 融合后分数：{len(fused_score)} 个")
        
        # Step4: 清洗
        fused_score = self.clean_keywords(fused_score)
        print(f"🔍 [DEBUG] 清洗后：{len(fused_score)} 个")
        
        # 如果没有关键词，使用备用方案
        if not fused_score:
            print("⚠️  [DEBUG] 未提取到关键词，尝试直接使用原始文本...")
            merged_text = " ".join(docs)
            words = jieba.lcut(merged_text)
            # 过滤停用词和单字
            filtered = [w for w in words if w not in self.stopwords and len(w) >= 2]
            if filtered:
                from collections import Counter
                word_freq = Counter(filtered)
                top_words = word_freq.most_common(res_top_n)
                print(f"✅ [DEBUG] 使用备用方案提取到 {len(top_words)} 个关键词")
                return [(w, f) for w, f in top_words]
            return []
        
        # Step5: 再次归一化并返回
        fused_score = self.normalize_scores(fused_score)
        result = sorted(fused_score.items(), key=lambda x: x[1], reverse=True)[:res_top_n]
        print(f"✅ [DEBUG] 最终返回 {len(result)} 个关键词\n")
        
        return result


# ======================
# example
# ======================
if __name__ == "__main__":
    docs = [
        "这个理财机器人挺方便的，但是风险提示有点模糊。",
        "整体还可以，不过有时候推荐太保守。",
        "我觉得这个AI理财助手不太透明。",
        "推荐逻辑很清楚，用着还行。",
        "风险解释不够清楚，让人有点不放心。"
    ]

    tkb = TextKeyBert()
    res = tkb.get_key_words(docs, 20)

    for r in res:
        print(r)
