#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R 语言热力图生成测试脚本
用于测试 rpy2 和 R 语言的 ggplot2 可视化功能
"""

import os
import sys
from r_heatmap_generator import RHeatmapGenerator
import numpy as np


def test_r_heatmap():
    """
    测试 R 语言热力图生成功能
    """
    print("=" * 60)
    print("开始测试 R 语言热力图生成")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    n_clusters = 5
    
    # 生成随机相似度矩阵（对角线为 1，其他值在 0-1 之间）
    similarity_matrix = np.random.rand(n_clusters, n_clusters) * 0.5 + 0.3
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # 对称矩阵
    np.fill_diagonal(similarity_matrix, 1.0)  # 对角线为 1
    
    # 创建聚类标签
    cluster_ids = list(range(1, n_clusters + 1))
    cluster_counts = {cid: np.random.randint(10, 100) for cid in cluster_ids}
    cluster_labels = [f"Cluster {cid}" for cid in cluster_ids]
    
    print(f"\n聚类数量：{n_clusters}")
    print(f"聚类 IDs: {cluster_ids}")
    print(f"聚类标签：{cluster_labels}")
    print(f"\n相似度矩阵形状：{similarity_matrix.shape}")
    print(f"相似度矩阵:\n{similarity_matrix}")
    
    # 生成 R 语言热力图
    try:
        generator = RHeatmapGenerator()
        save_path = "../heatmap/test_cluster_similarity_heatmap_R.png"
        
        print(f"\n正在生成 R 语言热力图...")
        result_path = generator.generate_heatmap(
            similarity_matrix=similarity_matrix,
            cluster_labels=cluster_labels,
            cluster_ids=cluster_ids,
            save_path=save_path
        )
        
        print(f"\n✅ 测试成功！")
        print(f"热力图已保存至：{result_path}")
        print(f"文件是否存在：{os.path.exists(result_path)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_r_heatmap()
    sys.exit(0 if success else 1)
