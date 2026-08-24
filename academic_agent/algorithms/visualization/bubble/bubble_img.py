import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sentence_transformers import SentenceTransformer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TopicVisualizer:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        # 修改模型基础路径为项目内的路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_base_dir = os.path.join(base_dir, '..', '..', '..', 'pretrain_models')
        self.df = None
        self.sentences = None
        self.cluster_ids = None
        self.cluster_counts = None
        self.cluster_labels = None
        self.embeddings = None
        self.cluster_centers = None
        self.cluster_coords = None

    # ---------------------------------------------------------
    # ① 加载 Excel 数据
    # ---------------------------------------------------------
    def load_data(self, sheet_name=None, content_column="content", cluster_column="cluster"):
        """
        加载聚类数据
        """
        # 如果没有指定sheet_name，则读取第一个工作表
        if sheet_name is None:
            excel_file = pd.ExcelFile(self.data_file_path)
            sheet_name = excel_file.sheet_names[0]  # 使用第一个工作表
            logger.info(f"未指定工作表，使用第一个工作表: {sheet_name}")
        
        self.df = pd.read_excel(self.data_file_path, sheet_name=sheet_name)

        # 覆盖列名（你之前的写法）
        self.df.columns = [content_column, cluster_column]

        self.sentences = self.df[content_column].tolist()

        # 聚类标签、数量
        self.cluster_ids = sorted(self.df[cluster_column].unique())
        self.cluster_counts = self.df[cluster_column].value_counts().sort_index().to_dict()

        # 标签名称：Cluster 1 (n=132)
        self.cluster_labels = [
            f"Cluster {cid} (n={self.cluster_counts[cid]})"
            for cid in self.cluster_ids
        ]

        logger.info(f"Data Loaded. Total sentences: {len(self.sentences)}")
        logger.info(f"Clusters: {self.cluster_ids}")

    # ---------------------------------------------------------
    # ② 生成句向量（BGE 模型）
    # ---------------------------------------------------------
    def generate_embeddings(self):
        """
        使用BERT/BGE模型生成句向量
        """
        logger.info("Loading embedding model...")
        embedding_folder = os.path.join(self.model_base_dir, 'bge-cn')
        
        # 检查模型路径是否存在
        if not os.path.exists(embedding_folder):
            raise FileNotFoundError(f"模型路径不存在: {embedding_folder}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(embedding_folder, device=device)

        logger.info("Generating embeddings...")
        self.embeddings = model.encode(
            self.sentences,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        logger.info(f"Embedding shape: {self.embeddings.shape}")

    # ---------------------------------------------------------
    # ③ 计算每个 cluster 的主题中心向量
    # ---------------------------------------------------------
    def compute_cluster_centers(self):
        """
        根据每个 cluster 的句向量计算主题中心向量（topic embedding）
        """
        centers = []

        for cid in self.cluster_ids:
            cluster_vecs = self.embeddings[self.df['cluster'] == cid]
            center = cluster_vecs.mean(axis=0)
            centers.append(center)

        self.cluster_centers = np.vstack(centers)
        logger.info(f"Cluster center shape: {self.cluster_centers.shape}")

    # ---------------------------------------------------------
    # ④ MDS 降维到 2D（类似 BERTopic 的 Intertopic Map）
    # ---------------------------------------------------------
    def reduce_dimensions(self):
        """
        将 cluster 中心向量降维到 2D
        """
        logger.info("Performing MDS dimensionality reduction...")
        mds = MDS(n_components=2, random_state=42)
        self.cluster_coords = mds.fit_transform(self.cluster_centers)

        logger.info(f"Coordinates shape: {self.cluster_coords.shape}")

    # ---------------------------------------------------------
    # ⑤ 绘制 Intertopic Distance Map 气泡图
    # ---------------------------------------------------------
    def plot_intertopic_map(self, save_path="intertopic_map.png"):
        """
        绘制论文风格的气泡图（Intertopic Distance Map）
        """
        logger.info("Plotting Intertopic Distance Map...")

        coords = self.cluster_coords
        sizes = np.array([self.cluster_counts[cid] for cid in self.cluster_ids])

        # 调整气泡大小，论文更清晰
        sizes = sizes / sizes.max() * 2500

        plt.figure(figsize=(9, 9))
        plt.scatter(
            coords[:, 0], coords[:, 1],
            s=sizes, alpha=0.45,
            edgecolor='black'
        )

        # 添加"Cluster 1 (n=132)"标签
        for i, cid in enumerate(self.cluster_ids):
            plt.text(
                coords[i, 0], coords[i, 1],
                self.cluster_labels[i],
                fontsize=10,
                ha='center', va='center'
            )

        plt.title("", fontsize=16)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

        logger.info(f"Figure saved to: {save_path}")


# ---------------------------------------------------------
# ★ 使用示例（你直接跑即可）
# ---------------------------------------------------------
if __name__ == "__main__":
    # 修改模型路径引用为相对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_base_dir = os.path.join(base_dir, '..', '..', '..', 'pretrain_models')
    embedding_model = os.path.join(model_base_dir, 'bge-cn')
    visualizer = TopicVisualizer(
        data_file_path="../../../data/c_data/f_llm_cluster_rd.xlsx"
    )

    visualizer.load_data()
    visualizer.generate_embeddings()
    visualizer.compute_cluster_centers()
    visualizer.reduce_dimensions()
    visualizer.plot_intertopic_map()