from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from data_preprocess import Data_PreProcessor
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextKeyBert(object):
    """A class for extracting keywords using BERT and LDA models."""

    def __init__(self):
        """Initialize the TextKeyBert with necessary components and parameters."""

        # Initialize data preprocessor for text cleaning and tokenization
        self.data_processor = Data_PreProcessor()

        # Create a custom tokenizer using the data processor's tokenizer
        self.custom_tokenizer = lambda text: self.data_processor.custom_tokenizer(text)

        # Initialize CountVectorizer with the custom tokenizer for text vectorization
        self.vectorizer = CountVectorizer(tokenizer=self.custom_tokenizer, token_pattern=None)

        # Path configuration for the Chinese embedding model
        model_base_dir = '/Users/wangchuixin/modelrepo/pretrainModels/aigc/'
        self.embedding_model = os.path.join(model_base_dir, 'bge-cn')

        # Initialize KeyBERT model with the pre-trained BERT model
        self.kbert_model = KeyBERT(model=self.embedding_model)

        # Parameters controlling the number of results to recall from each method
        self.lda_sentence_recall_number = 10        # Number of LDA keywords to recall per sentence
        self.keybert_sentence_recall_number = 10    # Number of KeyBERT keywords to recall per sentence
        self.keybert_doc_recall_number = 30         # Number of KeyBERT keywords to recall per document

        # Weight parameters for combining results from different methods
        self.lda_weight = 0.7               # Weight for LDA keywords in final combination
        self.keybert_sentence_weight = 0.7  # Weight for sentence-level KeyBERT keywords
        self.keybert_doc_weight = 0.2       # Weight for document-level KeyBERT keywords

    def normalize_scores(self, score_dict: dict) -> dict:
        if not score_dict:
            return {}
        min_score = float(min(score_dict.values()))
        max_score = float(max(score_dict.values()))
        if max_score == min_score:
            return {k: 1.0 for k in score_dict}
        return {k: float(v - min_score) / float(max_score - min_score) for k, v in score_dict.items()}

    def safe_extract_keywords(self, text, recall_topn=10, ngram_range=(2, 5), diversity=0.5):
        """Wrapper for KeyBERT.extract_keywords with error handling."""
        if not text or not text.strip():
            print(f"[WARN] Skip empty text: {repr(text)}")
            return []
        try:
            return self.kbert_model.extract_keywords(
                text,
                vectorizer=self.vectorizer,
                candidates=[],
                top_n=recall_topn,
                keyphrase_ngram_range=ngram_range,
                use_mmr=True,
                diversity=diversity
            )
        except ValueError as e:
            print(f"[WARN] extract_keywords failed: {e}, text={text[:50]}")
            return []

    def get_lda_res_dict(self, merged_text, recall_topn=10):
        if not merged_text.strip():
            return {}
        cntTf = self.vectorizer.fit_transform([merged_text])
        lda = LatentDirichletAllocation(n_components=5, max_iter=100,
                                        learning_method='batch')
        lda.fit(cntTf)
        tf_feature_names = self.vectorizer.get_feature_names_out()
        doc_topic_dist = lda.transform(cntTf)
        main_topic_id = doc_topic_dist[0].argmax()

        # 获取该主题下的 top-n 词汇
        topic_word_probs = lda.components_[main_topic_id]
        top_word_indices = topic_word_probs.argsort()[::-1][:recall_topn]
        lda_score_dict = {
            tf_feature_names[i]: topic_word_probs[i]
            for i in top_word_indices
        }
        return lda_score_dict

    def get_doc_keybert_res_dict(self, merged_text, recall_topn=10):
        keybert_recall_keywords_tuple = self.safe_extract_keywords(
            merged_text, recall_topn=recall_topn, ngram_range=(2, 10), diversity=0.8
        )
        result = {}
        for kw, score in keybert_recall_keywords_tuple:
            try:
                result[kw] = float(score) if not isinstance(score, str) else 0.0
            except Exception:
                result[kw] = 0.0
        return result

    def get_multi_sentence_keybert_res_dict(self, docs, recall_topn=10):
        keyword_score_dict = defaultdict(float)
        for doc in docs:
            keywords_in_one_doc = self.safe_extract_keywords(
                doc, recall_topn=recall_topn, ngram_range=(2, 8), diversity=0.3
            )
            for kw, score in keywords_in_one_doc:
                try:
                    if isinstance(score, str):
                        score = float(score)
                except Exception:
                    score = 0.0
                keyword_score_dict[kw] += score
        sorted_keywords = sorted(keyword_score_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_keywords[:30])

    def get_key_words(self, docs, extract_method, res_top_n=35):
        merged_text = " ".join(docs)
        lda_score_dict = {}

        # lda 限定候选词
        if extract_method == "lda":
            lda_score_dict = self.get_lda_res_dict(merged_text, self.lda_sentence_recall_number)

        # sentence-level抽取关键词
        keybert_multi_sentence_score_dict = self.get_multi_sentence_keybert_res_dict(
            docs, recall_topn=self.keybert_sentence_recall_number
        )
        # doc-level抽取关键词
        keybert_doc_score_dict = self.get_doc_keybert_res_dict(
            merged_text, recall_topn=self.keybert_doc_recall_number
        )

        # normalize
        lda_score_dict = self.normalize_scores(lda_score_dict)
        keybert_multi_sentence_score_dict = self.normalize_scores(keybert_multi_sentence_score_dict)
        keybert_doc_score_dict = self.normalize_scores(keybert_doc_score_dict)

        print("lda_score_dict is " + str(lda_score_dict))
        print("keybert_multi_sentence_score_dict is " + str(keybert_multi_sentence_score_dict))
        print("keybert_doc_score_dict is " + str(keybert_doc_score_dict))

        fused_score_dict = defaultdict(float)
        for kw, score in lda_score_dict.items():
            fused_score_dict[kw] += self.lda_weight * score

        for kw, score in keybert_multi_sentence_score_dict.items():
            fused_score_dict[kw] += self.keybert_sentence_weight * score

        for kw, score in keybert_doc_score_dict.items():
            fused_score_dict[kw] += self.keybert_doc_weight * score

        sorted_keywords = sorted(fused_score_dict.items(), key=lambda x: x[1], reverse=True)[:res_top_n]
        return sorted_keywords


# Example usage:
# docs = [
#     "空军大兄弟们很显然憋了一个大的，从作战体系讲，别说巴基斯坦陆军了，我认为解放军陆军一样跟不上空军这个节奏。",
#     "中国的科技七姐妹必有腾讯一席",
#     "腾讯是中国互联网企业的常青树"
# ]
# aa = TextKeyBert()
# res = aa.get_key_words(docs, "lda", 20)
# print(res)
