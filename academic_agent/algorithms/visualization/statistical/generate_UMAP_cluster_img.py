from academic_agent.algorithms.text_mining.embeddings_models.textEmbedding import TextEmbedding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from PIL import Image, ImageFilter, ImageDraw
import matplotlib



# --------------------------------------------------------
#           色彩增强函数（亮度、饱和度）
# --------------------------------------------------------
def darken_color(hex_color, factor=0.80):
    """点颜色使用稍微深一点（更清晰）"""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple((rgb * factor).clip(0, 1))

def lighten_color(hex_color, factor=1.15):
    """阴影颜色变浅一点（避免太深）"""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(np.minimum(rgb * factor, 1.0))

def saturate_color(hex_color, factor=1.40):
    """增强饱和度，使阴影色更加彩色"""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[1] = min(hsv[1] * factor, 1.0)  # S 饱和度增强
    new_rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return tuple(new_rgb.clip(0, 1))

def rgb255(c):
    """0-1 RGB → 0-255"""
    return tuple(int(v*255) for v in c)


# ========================================================
#                 主函数：彩色柔光阴影版（修改版）
# ========================================================
def generate_umap_seaborn_figure(data_path):

    # 1. 读数据
    text_embbdinger = TextEmbedding()
    df = pd.read_excel(data_path)

    # 检查数据列是否存在，如果不存在则使用前两列
    if 'content' in df.columns and 'cluster' in df.columns:
        texts = df['content'].tolist()
        labels = [int(l) for l in df['cluster'].tolist()]
    else:
        # 假设第一列是文本，第二列是聚类标签
        texts = df.iloc[:, 0].tolist()
        labels = [int(l) for l in df.iloc[:, 1].tolist()]

   
    # 动态生成topic_names，有多少个labels就有多少个topic
    clusters = sorted(set(labels))
    topic_names = {cid: f"topic {cid}" for cid in clusters}

    # 3. UMAP 降维
    embeddings = text_embbdinger.get_query_vector(texts)
    X_umap = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.10, random_state=42
    ).fit_transform(embeddings)

    plot_df = pd.DataFrame({
        "UMAP1": X_umap[:, 0],
        "UMAP2": X_umap[:, 1],
        "Cluster": labels
    })

    clusters = sorted(set(labels))


    # ======================================================
    #               亮彩点颜色（鲜艳不灰）
    # ======================================================
    bright_colors = [
        "#7A6FF0",  # violet
        "#FF7FB0",  # pink
        "#FFAA66",  # orange
        "#A78BFA",  # lavender
        "#FF9EC7",  # candy pink
        "#8FD8C8",  # mint
        "#7EC8FF",  # sky blue
        "#C58BFF",  # purple
        "#6FE3C8",  # aqua green
        "#FFB39E",  # peach
        "#9DE1A3",  # fresh green
    ][:len(clusters)]

    # 点颜色（鲜艳，无灰度）
    point_color_map = {cid: bright_colors[i] for i, cid in enumerate(clusters)}

    # ======================================================
    #   阴影颜色 = 亮彩颜色 → 变浅 → 增强饱和度（最关键步骤）
    # ======================================================
    cloud_color_map = {}
    for cid in clusters:
        base = bright_colors[cid - 1]

        # ① 稍微变浅一点
        c1 = lighten_color(base, factor=1.15)

        # ② 增强饱和度（让阴影更彩、更明显）
        c2 = saturate_color(c1, factor=1.40)

        cloud_color_map[cid] = c2


    # ======================================================
    #                     绘图
    # ======================================================
    fig, ax = plt.subplots(figsize=(10, 10))

    # 大画布，用于光晕绘制
    W, H = 3000, 3000

    # ======================================================
    #               4. 为每个簇绘制彩色柔光阴影
    # ======================================================
    for cid in clusters:

        sub = plot_df[plot_df["Cluster"] == cid]
        if len(sub) == 0:
            continue

        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        xnorm = (sub["UMAP1"] - plot_df["UMAP1"].min()) / (plot_df["UMAP1"].max() - plot_df["UMAP1"].min())
        ynorm = (sub["UMAP2"] - plot_df["UMAP2"].min()) / (plot_df["UMAP2"].max() - plot_df["UMAP2"].min())

        shadow_rgb = rgb255(cloud_color_map[cid])

        # 大点光源（阴影不会灰）
        for x, y in zip(xnorm, ynorm):
            px = int(x * (W - 1))
            py = int((1 - y) * (H - 1))

            r = 50
            draw.ellipse(
                (px-r, py-r, px+r, py+r),
                fill=shadow_rgb + (105,)      # **关键：更高 alpha → 阴影更彩**
            )

        # 彩色柔光模糊（颜色不冲掉）
        img = img.filter(ImageFilter.GaussianBlur(radius=22))

        ax.imshow(
            img,
            extent=[
                plot_df["UMAP1"].min(), plot_df["UMAP1"].max(),
                plot_df["UMAP2"].min(), plot_df["UMAP2"].max()
            ],
            alpha=0.90,
            zorder=0
        )


    # ======================================================
    #                  绘制前景点（鲜艳）
    # ======================================================
    for cid in clusters:
        sub = plot_df[plot_df["Cluster"] == cid]
        ax.scatter(
            sub["UMAP1"], sub["UMAP2"],
            s=22,
            color=point_color_map[cid],
            alpha=0.95,
            linewidth=0,
            zorder=5
        )

    # ======================================================
    #                在簇中心显示 c1, c2 等标签
    # ======================================================
    centers = plot_df.groupby("Cluster")[["UMAP1", "UMAP2"]].mean()
    
    for cid, (cx, cy) in centers.iterrows():
        ax.text(
            cx, cy,
            f"c{cid}",
            fontsize=12,
            weight="bold",
            color="black",  # 改为黑色字体
            ha="center",
            va="center",
            zorder=10
        )

    # ======================================================
    #                   标签绘制（左上角显示）
    # ======================================================
    # 获取图的边界
    x_min, x_max = plot_df["UMAP1"].min(), plot_df["UMAP1"].max()
    y_min, y_max = plot_df["UMAP2"].min(), plot_df["UMAP2"].max()
    
    # 计算左上角位置
    left_x = x_min + (x_max - x_min) * 0.05  # 左边 5% 的位置
    top_y = y_max - (y_max - y_min) * 0.05    # 上边 5% 的位置
    
    # 为每个簇添加标签到左上角
    for i, cid in enumerate(clusters):
        # 计算每个标签的y位置，使它们垂直排列
        label_y = top_y - (i * (y_max - y_min) * 0.03)  # 每个标签间隔3%
        
        # 格式化标签文本
        label_text = f"c{cid} = {topic_names[cid]}"
        
        ax.text(
            left_x, label_y,
            label_text,
            fontsize=8,
            weight="bold",
            color="black",
            ha="left",  # 水平左对齐
            va="center",  # 垂直居中
            family="Times New Roman",  # 使用Times New Roman字体
            zorder=10
        )

    # ======================================================
    #                   去掉坐标轴
    # ======================================================
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig("umap_cluster_supercolor_glow_modified.png", dpi=480, bbox_inches="tight")
    plt.show()