import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class ConfusionMatrixGenerator:
    """
    混淆矩阵生成器类
    负责生成和保存聚类相似度的混淆矩阵可视化图表
    """

    def __init__(self, similarity_matrix, cluster_labels, cluster_ids):
        """
        初始化混淆矩阵生成器

        参数:
        similarity_matrix (np.array): 聚类相似度矩阵
        cluster_labels (list): 聚类标签列表
        cluster_ids (list): 聚类ID列表
        """
        self.similarity_matrix = similarity_matrix
        self.cluster_labels = cluster_labels
        self.cluster_ids = cluster_ids

    def generate_heatmap(self, save_path="./cluster_similarity_heatmap_academic_caption.png"):
        """
        生成聚类相似度热力图

        参数:
        save_path (str): 保存路径
        """
        sns.set_theme(style="white")
        plt.figure(figsize=(11, 9))

        ax = sns.heatmap(
            self.similarity_matrix,
            cmap="viridis",
            annot=False,
            xticklabels=self.cluster_labels,
            yticklabels=self.cluster_labels,
            cbar_kws={"shrink": 0.55, "pad": 0.015},
            linewidths=0.25,
            linecolor="#e5e5e5"
        )

        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.title("Cluster Similarity Heatmap", fontsize=15, pad=18)

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()

        caption = (
            "Figure X. Heatmap of cosine similarity between clusters. "
            "Cluster labels include sample counts (n). "
            "Yellow = higher similarity; dark green = lower similarity."
        )

        plt.gcf().text(
            0.5, -0.07, caption,
            ha='center', va='center',
            fontsize=11
        )

        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"图已保存：{save_path}")
        
        return save_path