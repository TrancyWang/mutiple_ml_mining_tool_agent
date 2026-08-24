import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """
    聚类数据可视化类
    """

    def __init__(self, clusters, counts):
        """
        初始化聚类可视化器

        参数:
        clusters (list): 聚类标签列表
        counts (list): 各聚类的数量列表
        """
        self.clusters = clusters
        self.counts = counts

    def create_bar_chart(self, save_path="cluster_bar_sorted_desc.png"):
        """
        创建按数量从高到低排序（最多在最上方）的学术风横向条形图
        """

        clusters = self.clusters
        counts = self.counts

        # ---- 排序（从大到小），关键部分 ----
        sorted_idx = np.argsort(counts)[::-1]  # 降序
        clusters_sorted = [clusters[i] for i in sorted_idx]
        counts_sorted = [counts[i] for i in sorted_idx]

        # ---- 学术风主题 ----
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.2)

        # 柔和学术色
        colors = sns.color_palette("Set2", len(clusters_sorted))

        # ---- 横向条形图 ----
        plt.figure(figsize=(10, 7))

        bars = plt.barh(
            y=range(len(clusters_sorted)),
            width=counts_sorted,
            color=colors,
            alpha=0.9
        )

        # ---- 数字标注 ----
        for i, bar in enumerate(bars):
            plt.text(
                x=bar.get_width() + max(counts_sorted) * 0.02,
                y=bar.get_y() + bar.get_height() / 2,
                s=str(counts_sorted[i]),
                ha='left',
                va='center',
                fontsize=11,
                fontweight='bold',
                color="#333333"
            )

        # ---- Y 轴标签（保持顺序，从大到小）----
        plt.yticks(
            range(len(clusters_sorted)),
            [f"Cluster {c}" for c in clusters_sorted],
            fontsize=11
        )

        plt.xlabel("Number of Texts", fontsize=10)
        plt.title("Topic quantity ranking", fontsize=14, weight='bold')

        # ---- 学术风去脊线 ----
        sns.despine(top=True, right=True)

        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()

    def create_pie_chart(self, save_path="11 Cluster pie statistical.png"):
        """
        创建并显示聚类数据饼图

        参数:
        save_path (str): 保存图片的路径
        """
        # 避免标签过多时饼图难以阅读
        if len(self.clusters) > 15:
            logger.info("类别过多，跳过饼图显示")
            return

        plt.figure(figsize=(8, 8))
        plt.pie(self.counts, labels=self.clusters, autopct='%1.1f%%', startangle=90)
        plt.title('Percentage Distribution of Texts in Clusters')
        plt.axis('equal')
        plt.savefig(save_path)
        plt.show()