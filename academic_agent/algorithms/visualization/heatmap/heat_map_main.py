import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# 导入新创建的模块
from .cluster_similarity_analyzer import ClusterSimilarityAnalyzer
from .confusion_matrix_generator import ConfusionMatrixGenerator
from .semantic_super_cluster import SemanticSuperCluster


def generate_heatmap_chart(input_file, output_dir='./heatmap_results', model_base_dir=None):
    """
    生成热力图及相关分析结果
    
    参数:
    input_file (str): 输入数据文件路径（Excel 格式）
    output_dir (str): 结果输出目录
    model_base_dir (str): 预训练模型目录，默认为 None 时使用默认路径
    
    返回:
    str: 热力图保存路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置模型基础目录
    if model_base_dir is None:
        # 使用当前文件所在目录的 pretrain_models 文件夹
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        model_base_dir = os.path.join(project_root, 'pretrain_models')
    
    # =======================
    # 1. 读取数据
    # =======================
    analyzer = ClusterSimilarityAnalyzer(input_file, model_base_dir=model_base_dir)
    analyzer.load_data()
    sentences = analyzer.sentences

    # =======================
    # 2. BERT / BGE embedding
    # =======================
    analyzer.generate_embeddings()

    # =======================
    # 3. 聚合 cluster embedding
    # =======================
    analyzer.aggregate_cluster_embeddings()

    # =======================
    # 4. cluster 相似度矩阵
    # =======================
    analyzer.compute_similarity_matrix()

    # =======================
    # 4.1 打印 pairwise similarity (修改为混淆矩阵格式)
    # =======================
    confusion_df = analyzer.get_confusion_matrix_dataframe()
    confusion_csv_path = os.path.join(output_dir, "pairwise_cluster_similarity.csv")
    confusion_df.to_csv(confusion_csv_path, index=True)

    print("\n===== 所有 Pairwise Cluster Similarity 已保存 =====")
    print(f"{confusion_csv_path}\n")

    # =======================
    # 4.2 获取聚类对相似度
    # =======================
    pair_df = analyzer.get_pairwise_similarity_dataframe()

    # ===============================================
    # 4.3 自动找出 Top-k 最相似 / 最不相似的 pair（新增）
    # ===============================================
    TOP_K = 5

    top_k_sim = pair_df.sort_values("Similarity", ascending=False).head(TOP_K)
    bottom_k_sim = pair_df.sort_values("Similarity", ascending=True).head(TOP_K)

    print("===== Top-5 最相似的 Cluster Pair =====")
    print(top_k_sim, "\n")

    print("===== Top-5 最不相似的 Cluster Pair =====")
    print(bottom_k_sim, "\n")

    # 保存
    top_k_csv_path = os.path.join(output_dir, "top_k_most_similar_clusters.csv")
    bottom_k_csv_path = os.path.join(output_dir, "top_k_least_similar_clusters.csv")
    top_k_sim.to_csv(top_k_csv_path, index=False)
    bottom_k_sim.to_csv(bottom_k_csv_path, index=False)

    # ===============================================
    # 4.4 根据相似度自动聚类（语义超级簇）(新增)
    # ===============================================
    super_cluster_analyzer = SemanticSuperCluster(
        analyzer.similarity_matrix, 
        analyzer.cluster_ids, 
        analyzer.cluster_labels
    )
    super_cluster_analyzer.compute_distance_matrix()

    # 语义超级簇数量（你可以调 2~5）
    SUPER_K = 3

    super_cluster_analyzer.generate_super_clusters(n_clusters=SUPER_K)

    supercluster_df = super_cluster_analyzer.get_super_cluster_dataframe()
    supercluster_csv_path = os.path.join(output_dir, "semantic_superclusters.csv")
    supercluster_df.to_csv(supercluster_csv_path, index=False)

    print("===== 自动语义超级簇分组结果 =====")
    print(supercluster_df, "\n")

    # =======================
    # 5. 绘制论文版热力图（同之前）
    # =======================
    visualizer = ConfusionMatrixGenerator(
        analyzer.similarity_matrix,
        analyzer.cluster_labels,
        analyzer.cluster_ids
    )
    
    heatmap_save_path = os.path.join(output_dir, "cluster_similarity_heatmap_academic_caption.png")
    save_path = visualizer.generate_heatmap(heatmap_save_path)
    
    print(f"\n热力图已保存至：{save_path}")
    
    return save_path
