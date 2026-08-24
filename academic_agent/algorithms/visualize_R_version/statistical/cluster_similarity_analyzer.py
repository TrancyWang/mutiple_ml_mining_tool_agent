import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import sys
import re

# 获取项目根目录
def get_project_root():
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上追溯到项目根目录 (offline-algorithm/text_mining_tools)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return project_root

# 获取预训练模型目录
def get_model_dir():
    project_root = get_project_root()
    model_dir = os.path.join(project_root, 'pretrain_models')
    return model_dir


class ClusterSimilarityAnalyzer:
    """
    聚类相似度分析器类
    负责处理文本嵌入、计算聚类相似度等功能
    """

    def __init__(self, data_file_path, model_base_dir=None): # 使用项目根目录计算模型路径
        if model_base_dir is None:
            model_base_dir = get_model_dir()
        """
        初始化聚类相似度分析器

        参数:
        data_file_path (str): 数据文件路径
        model_base_dir (str): 预训练模型基础目录，默认使用项目根目录计算得到的路径
        """
        self.data_file_path = data_file_path
        self.model_base_dir = model_base_dir
        self.df = None
        self.sentences = None
        self.cluster_ids = None
        self.cluster_counts = None
        self.cluster_labels = None
        self.embeddings = None
        self.cluster_vectors = None
        self.similarity_matrix = None

    def load_data(self, sheet_name=None, content_column=None, cluster_column="cluster", theme_column=None):
        """
        加载聚类数据

        参数:
        sheet_name (str): Excel工作表名称，如果为None则自动选择第一个工作表
        content_column (str): 内容列名（可选，热力图不强制需要，用于生成文本嵌入）
        cluster_column (str): 聚类标签列名
        theme_column (str): 主题名称列名（可选），用于生成人类可读的类别名，如 "en_theme"
        """
        if sheet_name is None:
            # 如果未指定工作表名称，则读取第一个工作表
            excel_file = pd.ExcelFile(self.data_file_path)
            sheet_name = excel_file.sheet_names[0] if excel_file.sheet_names else 0
            
        self.df = pd.read_excel(self.data_file_path, sheet_name=sheet_name)
        
        # 自动识别文本列（优先参数值，否则常见列名）
        if content_column and content_column in self.df.columns:
            self.content_column = content_column
        else:
            candidates = ["text", "content", "sentence", "review", "comment", "摘要", "文本", "评论"]
            self.content_column = next(
                (c for c in candidates if c in self.df.columns), None
            )
        
        # 聚类列
        if cluster_column not in self.df.columns:
            cluster_candidates = ["cluster", "bert_cluster", "topic", "label", "group", "类别", "clustering"]
            cluster_column = next(
                (c for c in cluster_candidates if c in self.df.columns), self.df.columns[-1]
            )
        self.cluster_column = cluster_column
        self.theme_column = theme_column
        
        # 验证聚类列存在
        if self.cluster_column not in self.df.columns:
            raise ValueError(f"聚类列 '{self.cluster_column}' 不存在于数据中，可用列：{list(self.df.columns)}")
        
        # 文本列可选：用于生成嵌入，缺失则用空串占位
        if self.content_column and self.content_column in self.df.columns:
            self.sentences = self.df[self.content_column].astype(str).tolist()
        else:
            print("⚠️ 未找到文本列，将使用空字符串占位（热力图相似度将退化为基于类别一致性）")
            self.sentences = [""] * len(self.df)
        
        self.cluster_ids = sorted(self.df[cluster_column].unique())
        self.cluster_counts = self.df[cluster_column].value_counts().sort_index().to_dict()
        
        # 构建主题名称映射
        theme_map = {}
        if theme_column and theme_column in self.df.columns:
            for cid in self.cluster_ids:
                mask = self.df[cluster_column] == cid
                theme_vals = self.df.loc[mask, theme_column].dropna()
                if len(theme_vals) > 0:
                    theme_map[cid] = str(theme_vals.iloc[0])
            print(f"📝 已从列 '{theme_column}' 加载主题名称映射: {theme_map}")
        self.theme_map = theme_map
        
        # ⭐ 坐标标签：与柱状图一致，只显示 cluster 本身
        #    规则：字母+数字（如 R1）直接用；纯数字（如 1,2,3）加前缀 c -> c1,c2
        def format_cluster_display(cid):
            str_cid = str(cid)
            if re.fullmatch(r'-?\d+(\.\d+)?', str_cid):
                return f"c{str_cid}"
            return str_cid
        
        self.cluster_labels = [format_cluster_display(cid) for cid in self.cluster_ids]
        
        # ⭐ 底部说明文字：把 en_theme 写在最下面（一行一个 cid -> 主题名）
        #    若主题名包含中文等非 ASCII 字符，则不写入图片（避免热力图出现中文）
        def _is_ascii(text):
            return all(ord(ch) < 128 for ch in str(text))

        theme_lines = []
        for cid in self.cluster_ids:
            disp = format_cluster_display(cid)
            if cid in theme_map and _is_ascii(theme_map[cid]):
                theme_lines.append(f"{disp} = {theme_map[cid]}")
            else:
                theme_lines.append(f"{disp} = Cluster {cid}")
        self.theme_lines = theme_lines

    def generate_embeddings(self):
        """
        使用BERT/BGE模型生成文本嵌入向量
        """
        embedding_folder = os.path.join(self.model_base_dir, 'bge-cn')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(embedding_folder, device=device)
        self.embeddings = model.encode(self.sentences, convert_to_numpy=True, show_progress_bar=True)

    def aggregate_cluster_embeddings(self):
        """
        计算每个聚类的平均嵌入向量
        """
        cluster_vectors = []
        for cid in self.cluster_ids:
            idx = self.df[self.df[self.cluster_column] == cid].index
            cluster_emb = self.embeddings[idx].mean(axis=0)
            cluster_vectors.append(cluster_emb)
        self.cluster_vectors = np.vstack(cluster_vectors)

    def compute_similarity_matrix(self):
        """
        计算聚类间的相似度矩阵
        """
        self.similarity_matrix = cosine_similarity(self.cluster_vectors)
        
    def get_pairwise_similarity_dataframe(self):
        """
        获取聚类对之间的相似度DataFrame
        
        返回:
        pd.DataFrame: 包含聚类对及其相似度的DataFrame
        """
        pair_list = []
        for i, cid_i in enumerate(self.cluster_ids):
            for j, cid_j in enumerate(self.cluster_ids):
                if i < j:  # 只获取上三角部分（避免重复）
                    sim = self.similarity_matrix[i][j]
                    pair_list.append([cid_i, cid_j, sim])

        pair_df = pd.DataFrame(pair_list, columns=["Cluster_A", "Cluster_B", "Similarity"])
        return pair_df
        
    def get_confusion_matrix_dataframe(self):
        """
        获取聚类间相似度的混淆矩阵形式DataFrame
        
        返回:
        pd.DataFrame: 以混淆矩阵形式表示的聚类相似度DataFrame
        """
        # 创建一个完整的相似度矩阵（包括对角线）
        confusion_matrix = pd.DataFrame(
            self.similarity_matrix, 
            index=self.cluster_ids, 
            columns=self.cluster_ids
        )
        return confusion_matrix