import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
from pathlib import Path


class ClusterSimilarityAnalyzer:
    """
    聚类相似度分析器类
    负责处理文本嵌入、计算聚类相似度等功能
    """

    def __init__(self, data_file_path, model_base_dir=None):
        """
        初始化聚类相似度分析器

        参数:
        data_file_path (str): 数据文件路径
        model_base_dir (str): 预训练模型基础目录
        """
        self.data_file_path = data_file_path
        self.model_base_dir = model_base_dir or os.getenv(
            "PRETRAINED_MODELS_DIR",
            str(Path(__file__).resolve().parents[3] / "pretrain_models"),
        )
        self.df = None
        self.sentences = None
        self.cluster_ids = None
        self.cluster_counts = None
        self.cluster_labels = None
        self.embeddings = None
        self.cluster_vectors = None
        self.similarity_matrix = None

    def load_data(self, sheet_name="Sheet1", content_column="content", cluster_column="cluster"):
        """
        加载聚类数据

        参数:
        sheet_name (str): Excel工作表名称
        content_column (str): 内容列名
        cluster_column (str): 聚类标签列名
        """
        self.df = pd.read_excel(self.data_file_path, sheet_name=sheet_name)
        self.df.columns = [content_column, cluster_column]
        self.sentences = self.df[content_column].tolist()
        self.cluster_ids = sorted(self.df[cluster_column].unique())
        self.cluster_counts = self.df[cluster_column].value_counts().sort_index().to_dict()
        self.cluster_labels = [f"Cluster {cid} (n={self.cluster_counts[cid]})" for cid in self.cluster_ids]

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
            idx = self.df[self.df["cluster"] == cid].index
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
