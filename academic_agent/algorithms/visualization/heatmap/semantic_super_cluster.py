import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


class SemanticSuperCluster:
    """
    语义超级簇分析器类
    负责基于聚类相似度进行二次聚类，形成语义超级簇
    """

    def __init__(self, similarity_matrix, cluster_ids, cluster_labels):
        """
        初始化语义超级簇分析器

        参数:
        similarity_matrix (np.array): 聚类相似度矩阵
        cluster_ids (list): 聚类ID列表
        cluster_labels (list): 聚类标签列表
        """
        self.similarity_matrix = similarity_matrix
        self.cluster_ids = cluster_ids
        self.cluster_labels = cluster_labels
        self.distance_matrix = None
        self.super_cluster_labels = None

    def compute_distance_matrix(self):
        """
        根据相似度矩阵计算距离矩阵
        distance = 1 - similarity
        """
        self.distance_matrix = 1 - self.similarity_matrix

    def generate_super_clusters(self, n_clusters=3, metric='euclidean', linkage='ward'):
        """
        生成语义超级簇

        参数:
        n_clusters (int): 超级簇数量
        metric (str): 距离度量方式
        linkage (str): 聚类链接方式
        """
        if self.distance_matrix is None:
            self.compute_distance_matrix()

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage
        )

        self.super_cluster_labels = clustering.fit_predict(self.distance_matrix)

    def get_super_cluster_dataframe(self):
        """
        获取语义超级簇结果的DataFrame

        返回:
        pd.DataFrame: 包含聚类ID、标签和所属超级簇的DataFrame
        """
        if self.super_cluster_labels is None:
            raise ValueError("尚未生成语义超级簇，请先调用 generate_super_clusters 方法")

        supercluster_df = pd.DataFrame({
            "Cluster_ID": self.cluster_ids,
            "Cluster_Label": self.cluster_labels,
            "Supercluster": self.super_cluster_labels
        })
        
        return supercluster_df