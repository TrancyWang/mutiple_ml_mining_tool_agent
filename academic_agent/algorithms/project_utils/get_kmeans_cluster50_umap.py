import pandas as pd
import numpy as np
import os
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =======================
# 1. 读取数据
# =======================
# 使用绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
data_file = os.path.join(project_root, 'text_mining', 'data', 'llm_cluster_rd.csv')
df = pd.read_csv(data_file, sep="\t")  # 假设第一列是句子，第二列是已有的聚类标签
df.columns = ["content", "cluster"]

# 如果你只想要句子列表：
sentences = df["content"].tolist()

# =======================
# 2. 生成句子向量 (BERT / BGE)
# =======================
# 使用绝对路径计算模型目录
model_base_dir = os.path.join(project_root, 'pretrain_models')
embedding_model = os.path.join(model_base_dir, 'bge-cn')
model = SentenceTransformer(embedding_model)  # 换成你实际用的模型名称
embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)

# =======================
# 3. UMAP 降维
# =======================
umap_model = umap.UMAP(n_components=2, random_state=42)  # n_components可调
X_umap = umap_model.fit_transform(embeddings)

# =======================
# 4. KMeans 聚类（如果你需要重新聚类）
# =======================
# 如果你想直接用已有的 cluster 列，可以跳过这一步
n_clusters = len(df["cluster"].unique())  # 或者手动设定
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(X_umap)

# =======================
# 5. 计算簇中心 & 距离
# =======================
cluster_centers = []
for c in sorted(df["cluster"].unique()):
    cluster_centers.append(X_umap[df["cluster"] == c].mean(axis=0))
cluster_centers = np.vstack(cluster_centers)

df["distance_to_center"] = [
    np.linalg.norm(X_umap[i] - cluster_centers[df.loc[i, "cluster"]])
    for i in range(len(df))
]

# =======================
# 6. 每簇取最近的 50 条句子
# =======================
top50_each_cluster = (
    df.groupby("cluster", group_keys=False)
      .apply(lambda g: g.nsmallest(50, "distance_to_center"))
      .reset_index(drop=True)
)

# 调整列顺序：sentence, cluster, distance_to_center
top50_each_cluster = top50_each_cluster[["content", "cluster", "distance_to_center"]]

# =======================
# 7. 保存结果
# =======================
output_file = os.path.join(project_root, 'data', 'cluster_top50_center.csv')
top50_each_cluster.to_csv(output_file, index=False, encoding="utf-8-sig")

logger.info("✅ 已生成 cluster_top50_center.csv，每个簇心最近的50条句子已保存。")
