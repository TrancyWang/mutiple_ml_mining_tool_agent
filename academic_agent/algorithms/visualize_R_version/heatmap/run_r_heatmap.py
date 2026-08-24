#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 使用 R 语言生成热力图
从 test_clusters.xlsx 文件读取数据并生成 R 语言版本的热力图
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔥 Qt / GUI 稳定性关键（Intel Mac 必加）
os.environ["QT_MAC_WANTS_LAYER"] = "1"
os.environ["QT_QPA_PLATFORM"] = "cocoa"
import sys
import pandas as pd
import numpy as np
from r_heatmap_generator import RHeatmapGenerator
from src_code.text_mining_method.embeddings_models.textEmbedding import TextEmbedding
from sklearn.preprocessing import normalize

def main():
    """
    主函数：从 test_clusters.xlsx 生成 R 语言热力图
    """
    print("=" * 70)
    print("开始生成 R 语言聚类相似度热力图")
    print("=" * 70)
    
    # 检查输入文件是否存在
    input_file = "clusters.xlsx"
    if not os.path.exists(input_file):
        print(f"\n❌ 错误：找不到输入文件 '{input_file}'")
        print(f"请确保 {input_file} 文件在当前目录下")
        print("\n示例数据文件位置:")
        print(f"  {os.path.join(os.path.dirname(__file__), 'test_clusters.xlsx')}")
        return False
    
    # 创建输出目录
    output_dir = "output_r_heatmap"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 输入文件：{input_file}")
    print(f"📂 输出目录：{output_dir}")
    print("\n⏳ 正在处理中...\n")
    
    try:
        # 读取 test_clusters.xlsx 数据
        print(f"正在读取数据文件：{input_file}")
        df = pd.read_excel(input_file)
        print(f"数据列名: {df.columns.tolist()}")
        
        # 从数据中提取聚类信息
        text_col = 'Text' if 'Text' in df.columns else df.columns[0]
        cluster_col = 'Cluster' if 'Cluster' in df.columns else df.columns[1]

        
        # 检查列名是否匹配
        if cluster_col != 'Cluster':
            print(f"⚠️  警告：未找到 'Cluster' 列，使用第一列 '{cluster_col}' 作为聚类列")
        if text_col != 'Text':
            print(f"⚠️  警告：未找到 'Text' 列，使用第二列 '{text_col}' 作为文本列")
        print(f"使用的聚类列: {cluster_col}, 文本列: {text_col}")
        
        # 确保文本列为字符串类型，处理NaN值
        df[text_col] = df[text_col].fillna('').astype(str)
        
        cluster_ids = sorted(df[cluster_col].unique())
        sentences = df[text_col].tolist()
        
        print(f"共读取 {len(sentences)} 条文本")
        print(f"共 {len(cluster_ids)} 个聚类")
        print(f"前3个句子示例: {sentences[:3]}")
        # 确保所有句子都是字符串类型（双重检查）
        sentences = [str(s) for s in sentences]
        
        # 生成文本嵌入向量（使用 BGE 模型）
        print("\n正在生成文本嵌入向量...")

        # 使用 TextEmbedding 类
        embedder = TextEmbedding()
        embeddings = embedder.get_query_vector(sentences)
        
        if embeddings is None:
            print("❌ 错误：生成嵌入向量失败")
            return False
        
        print(f"嵌入向量形状: {embeddings.shape}")
        print(f"嵌入向量数据类型: {embeddings.dtype}")
        print(f"嵌入向量范数范围: {np.linalg.norm(embeddings, axis=1).min():.4f} - {np.linalg.norm(embeddings, axis=1).max():.4f}")
        
        # 聚合聚类嵌入向量
        print("正在聚合聚类嵌入向量...")
        cluster_vectors = []
        for cid in cluster_ids:
            mask = df[cluster_col] == cid
            cluster_vec = embeddings[mask].mean(axis=0)
            cluster_vectors.append(cluster_vec)
        cluster_vectors = np.array(cluster_vectors)
        print(f"聚类向量形状: {cluster_vectors.shape}")
        print(f"每个聚类的样本数: {df[cluster_col].value_counts().sort_index().to_dict()}")
        
        # 归一化
        cluster_vectors = normalize(cluster_vectors)
        print(f"归一化后聚类向量范数: {np.linalg.norm(cluster_vectors, axis=1)}")
        
        # 计算相似度矩阵
        print("正在计算聚类相似度矩阵...")
        similarity_matrix = np.dot(cluster_vectors, cluster_vectors.T)
        
        # 调试：打印相似度矩阵统计信息
        print(f"相似度矩阵形状: {similarity_matrix.shape}")
        print(f"相似度矩阵最小值: {similarity_matrix.min():.6f}")
        print(f"相似度矩阵最大值: {similarity_matrix.max():.6f}")
        print(f"相似度矩阵均值: {similarity_matrix.mean():.6f}")
        print(f"相似度矩阵对角线值: {np.diag(similarity_matrix)}")
        print(f"相似度矩阵是否对称: {np.allclose(similarity_matrix, similarity_matrix.T)}")
        print(f"聚类向量形状: {cluster_vectors.shape}")
        print(f"聚类向量L2范数: {np.linalg.norm(cluster_vectors, axis=1)}")
        
        # 创建聚类标签
        cluster_counts = df[cluster_col].value_counts().to_dict()
        cluster_labels = [f"Cluster {cid}" for cid in cluster_ids]
        
        # 生成 R 语言热力图
        generator = RHeatmapGenerator()
        save_path = os.path.join(output_dir, "cluster_similarity_heatmap_R.png")
        
        result_path = generator.generate_heatmap(
            similarity_matrix=similarity_matrix,
            cluster_labels=cluster_labels,
            cluster_ids=cluster_ids,
            save_path=save_path
        )
        
        print("\n" + "=" * 70)
        print("✅ 处理完成！")
        print("=" * 70)
        
        print(f"\n📊 生成的文件:")
        print(f"  {result_path}")
        
        print(f"\n💡 提示：使用以下命令查看图片:")
        print(f"  macOS:   open {result_path}")
        print(f"  Linux:   xdg-open {result_path}")
        print(f"  Windows: start {result_path}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ 处理失败：{e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
