#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Heatmap visualization for overall sentiment structure across clusters
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # =========================
    # 1. 读取数据
    # =========================
    file_path = "cluster_sentiment_matrix.xlsx"
    df = pd.read_excel(file_path,sheet_name="All")

    # =========================
    # 2. 基础校验
    # =========================
    required_cols = [
        "cluster",
        "Very Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Very Positive"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要字段: {col}")

    # =========================
    # 3. 设置 index & 列顺序
    # =========================
    df = df.set_index("cluster")

    sentiment_order = [
        "Very Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Very Positive"
    ]
    df = df[sentiment_order]

    # =========================
    # 4. 画图风格（论文友好）
    # =========================
    sns.set(
        style="white",
        font_scale=1.1
    )

    # =========================
    # 5. 热力图（数量版）
    # =========================
    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white"
    )

    plt.xlabel("Sentiment Category")
    plt.ylabel("Cluster")
    plt.title("Overall Sentiment Distribution Across Clusters")

    plt.tight_layout()
    plt.savefig("heatmap_sentiment_absolute.png", dpi=300)
    plt.show()

    # =========================
    # 6. 热力图（比例版｜推荐）
    # =========================
    df_ratio = df.div(df.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df_ratio,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white"
    )

    plt.xlabel("Sentiment Category")
    plt.ylabel("Cluster")
    plt.title("Normalized Sentiment Structure Across Clusters")

    plt.tight_layout()
    plt.savefig("heatmap_sentiment_normalized.png", dpi=300)
    plt.show()

    print("✅ 热力图绘制完成")
    print("📁 输出文件：")
    print(" - heatmap_sentiment_absolute.png")
    print(" - heatmap_sentiment_normalized.png")


if __name__ == "__main__":
    main()