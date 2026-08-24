import pandas as pd
import numpy as np

def analyze_cluster_similarities(file_path):
    """
    分析聚类相似度，找出每个聚类最相似的前5个聚类

    参数:
    csv_file_path (str): 相似度矩阵CSV文件路径
    """
    # 读取相似度矩阵
    # df = pd.read_csv(file_path, index_col=0)
    df = pd.read_excel(file_path, index_col=0)
    # 获取聚类ID
    cluster_ids = df.index.tolist()

    print("=" * 50)
    print("每个类别的前5个最相似的cluster:")
    print("=" * 50)

    # 存储所有相似度值用于后续分析
    all_similarities = []

    # 对每个聚类找出最相似的前5个聚类
    for cluster_id in cluster_ids:
        # 获取当前聚类与其他聚类的相似度
        # 注意：索引是整数，但列名是字符串
        similarities = df.loc[cluster_id].drop(str(cluster_id))  # 移除自己与自己的相似度

        # 按相似度降序排列
        top_5_similar = similarities.nlargest(5)

        print(f"\nCluster {cluster_id} 的前5个最相似聚类:")
        for i, (other_cluster, similarity) in enumerate(top_5_similar.items(), 1):
            print(f"  {i}. Cluster {other_cluster}: {similarity:.3f}")  # Changed to 3 decimal places
            # 记录相似度对（避免重复）
            if int(cluster_id) < int(other_cluster):
                all_similarities.append((int(cluster_id), int(other_cluster), similarity))

    # 找出整体最相似的聚类对
    print("\n" + "=" * 50)
    print("整体最相似的聚类对:")
    print("=" * 50)

    # 按相似度排序
    all_similarities.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 10 最相似的聚类对:")
    for i, (cluster_a, cluster_b, similarity) in enumerate(all_similarities[:10], 1):
        print(f"  {i}. Cluster {cluster_a} & Cluster {cluster_b}: {similarity:.3f}")  # Changed to 3 decimal places

    # 最相似的一对
    most_similar_pair = all_similarities[0]
    print(f"\n最相似的聚类对是: Cluster {most_similar_pair[0]} 和 Cluster {most_similar_pair[1]} "
          f"(相似度: {most_similar_pair[2]:.3f})")  # Changed to 3 decimal places

if __name__ == "__main__":
    # 分析聚类相似度
    analyze_cluster_similarities("../pairwise_cluster_similarity.xlsx")
