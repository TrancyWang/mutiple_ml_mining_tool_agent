# -*- coding: utf-8 -*-
"""
UMAP 聚类可视化生成器（R 版本）
功能：
1. 自动识别 Excel 中的 cluster 列（支持 cluster/topic/label/group 等名称）
2. 从 sheet2 读取 topic 名称映射
3. 统一转换为 Cluster 列名传递给 R
4. 保存到 data_analysis_result/text_mining_imgs 目录
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class UMAPClusterVisualizer:
    """UMAP 聚类可视化器"""

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        self.model_base_dir = os.path.join(project_root, 'pretrain_models')

        self.df = None
        self.sentences = None
        self.labels = None
        self.embeddings = None
        self.topic_names = None

    # ================================
    # ① 加载数据 + 自动识别列名
    # ================================
    def load_data(self, sheet_name=0, text_column=None, cluster_column=None, theme_column=None):
        """
        加载数据并自动识别 cluster 列

        Parameters:
        -----------
        sheet_name : int or str, optional
            Excel 工作表名称或索引，默认为第一个工作表 (0)
        text_column : str, optional
            文本内容列名，为 None 则自动识别
        cluster_column : str, optional
            聚类标签列名，为 None 则自动识别
        theme_column : str, optional
            主题名称列名（如 "en_theme"），用于在图中显示人类可读的类别名，
            优先于 sheet2 的映射表
        """
        if sheet_name is None:
            sheet_name = 0

        print(f"📂 正在读取 Excel 文件：{self.data_file_path}")
        print(f"📄 读取 Sheet: {sheet_name}")

        self.df = pd.read_excel(self.data_file_path, sheet_name=sheet_name)

        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"读取 Excel 失败，期望得到 DataFrame，但得到了 {type(self.df)}")

        print(f"📋 原始列名：{self.df.columns.tolist()}")
        print(f"📊 数据形状：{self.df.shape}")
        print(f"🔍 前 3 行数据预览:\n{self.df.head(3)}")

        # ===== 保存原始列名（用于 theme_column 查找） =====
        self.original_columns = list(self.df.columns)
        
        # ===== 统一列名：去除空格并转小写用于匹配，但保留原始列名引用 =====
        lower_to_orig = {col.strip().lower(): col for col in self.df.columns}
        self.df.columns = [col.strip().lower() for col in self.df.columns]
        print(f"📝 处理后列名（小写）：{self.df.columns.tolist()}")

        # ===== 识别 cluster 列 =====
        if cluster_column and cluster_column in self.df.columns:
            detected_cluster = cluster_column
        elif cluster_column and cluster_column.lower() in lower_to_orig:
            detected_cluster = cluster_column.lower()
        else:
            cluster_candidates = ["cluster", "topic", "label", "group", "类别", "clustering"]
            detected_cluster = None
            for col in self.df.columns:
                if col in cluster_candidates:
                    detected_cluster = col
                    break
            if detected_cluster is None:
                print("⚠️ 未找到标准 cluster 列名，尝试使用最后一列作为 cluster 列")
                detected_cluster = self.df.columns[-1]
                print(f"📍 使用最后一列作为 cluster 列：{detected_cluster}")
        print(f"✅ 检测到 cluster 列：{detected_cluster}")

        # ===== 识别文本列 =====
        if text_column and text_column in self.df.columns:
            detected_text = text_column
        elif text_column and text_column.lower() in lower_to_orig:
            detected_text = text_column.lower()
        else:
            text_candidates = ["sentence", "text", "content", "review", "comment", "文本", "评论"]
            detected_text = None
            for col in self.df.columns:
                if col in text_candidates:
                    detected_text = col
                    break
            if detected_text is None:
                print("⚠️ 未找到标准 text 列名，尝试使用第一列作为文本列")
                detected_text = self.df.columns[0]
                print(f"📍 使用第一列作为文本列：{detected_text}")
        print(f"✅ 检测到文本列：{detected_text}")

        # ===== 统一列名 =====
        self.df.rename(columns={
            detected_cluster: "Cluster",
            detected_text: "Sentence"
        }, inplace=True)

        print(f"✅ 标准化后列名：{self.df.columns.tolist()}")
        print(f"🔍 重命名后前 3 行:\n{self.df[['Cluster', 'Sentence']].head(3)}")

        # ===== 数据提取 =====
        self.sentences = self.df["Sentence"].tolist()
        self.labels = self.df["Cluster"].tolist()
        print(f"📊 提取了 {len(self.sentences)} 条文本，{len(set(self.labels))} 个聚类标签")

        # ===== 读取 topic 名称映射（优先 theme_column，其次 sheet2） =====
        self.topic_names = {}
        
        # 优先从 theme_column 构建映射
        if theme_column:
            theme_col = theme_column.lower() if theme_column.lower() in self.df.columns else theme_column
            if theme_col in self.df.columns:
                print(f"\n📖 正在从列 '{theme_column}' 读取主题名称映射...")
                for cid in self.df["Cluster"].unique():
                    mask = self.df["Cluster"] == cid
                    theme_vals = self.df.loc[mask, theme_col].dropna()
                    if len(theme_vals) > 0:
                        self.topic_names[cid] = str(theme_vals.iloc[0])
                print(f"✅ 从列 '{theme_column}' 加载 {len(self.topic_names)} 个主题名称")
                print(f"   Topic 映射：{self.topic_names}")
            else:
                print(f"⚠️ 列 '{theme_column}' 不存在，将尝试 sheet2")

        # 如果 theme_column 没加载到数据，回退到 sheet2
        if not self.topic_names:
            print("\n📖 正在从 sheet2 读取 topic 名称映射...")
            try:
                topic_df = pd.read_excel(self.data_file_path, sheet_name='sheet2')
                print(f"📋 sheet2 列名：{topic_df.columns.tolist()}")
                print(f"🔍 sheet2 前 3 行:\n{topic_df.head(3)}")
                self.topic_names = dict(zip(topic_df.iloc[:, 0], topic_df.iloc[:, 1]))
                print(f"✅ 从 sheet2 成功加载 {len(self.topic_names)} 个主题名称映射")
                print(f"   Topic 映射：{self.topic_names}")
            except Exception as e:
                print(f"⚠️ 无法从 sheet2 读取主题名称：{e}")
                print("📝 将使用默认命名规则（Cluster + 编号）")
                self.topic_names = {}

    # ================================
    # ② embeddings
    # ================================
    def generate_embeddings(self):
        """使用 BGE 模型生成文本嵌入"""
        embedding_folder = os.path.join(self.model_base_dir, 'bge-cn')
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = SentenceTransformer(embedding_folder, device=device)

        self.embeddings = model.encode(
            self.sentences,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        print(f"✅ 生成 {len(self.embeddings)} 个嵌入向量")

    # ================================
    # ③ UMAP 降维
    # ================================
    def reduce_dimensions(self, n_components=2, n_neighbors=5, min_dist=0.00, spread=2.5, random_state=42):
        """
        使用 UMAP 进行降维

        Parameters:
        -----------
        n_components : int, optional
            降维后的维度数，默认 2
        n_neighbors : int, optional
            UMAP 邻居数，默认 12
        min_dist : float, optional
            最小距离，默认 0.55
        spread : float, optional
            扩散参数，默认 2.5
        random_state : int, optional
            随机种子，默认 42
        """
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            random_state=random_state
        )
        X_umap = reducer.fit_transform(self.embeddings)

        plot_df = pd.DataFrame({
            "UMAP1": X_umap[:, 0],
            "UMAP2": X_umap[:, 1],
            "Cluster": self.labels
        })

        # 拉开 cluster（增强分离）
        centers = plot_df.groupby("Cluster")[["UMAP1", "UMAP2"]].mean()

        expanded = []
        for cid in plot_df["Cluster"].unique():
            sub = plot_df[plot_df["Cluster"] == cid].copy()
            cx, cy = centers.loc[cid]

            scale = 2.3
            sub["UMAP1"] = cx + (sub["UMAP1"] - cx) * scale
            sub["UMAP2"] = cy + (sub["UMAP2"] - cy) * scale

            expanded.append(sub)

        self.plot_df = pd.concat(expanded, ignore_index=True)
        print(f"✅ UMAP 降维完成，数据形状：{self.plot_df.shape}")

    # ================================
    # ④ R 绘图
    # ================================
    # ================================
    # ④ R 绘图
    # ================================
    def plot_with_r(self, save_filename="umap_cluster_R.png", save_dir=None, style="density"):

        # 优先使用用户指定的保存目录，否则回退默认目录
        if save_dir and str(save_dir).strip():
            img_save_dir = str(save_dir).strip()
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            img_save_dir = os.path.join(project_root, 'data_analysis_result', 'text_mining_imgs')

        os.makedirs(img_save_dir, exist_ok=True)
        image_path = os.path.join(img_save_dir, save_filename)

        print(f"📋 Python plot_df 列名：{self.plot_df.columns.tolist()}")

        from rpy2.robjects import default_converter

        self.plot_df.columns = [str(col) for col in self.plot_df.columns]

        with localconverter(default_converter + pandas2ri.converter):
            robjects.globalenv["plot_df"] = self.plot_df

        if not self.topic_names:
            cluster_ids = sorted(self.plot_df["Cluster"].unique())
            self.topic_names = {cid: f"Cluster {cid}" for cid in cluster_ids}

        topic_r = robjects.ListVector({str(k): v for k, v in self.topic_names.items()})
        robjects.globalenv["topic_names"] = topic_r

        # ===============================
        # 🔥 这里是唯一修改的部分（R代码）
        # ===============================
        r_code = f"""
        library(ggplot2)
        library(dplyr)

        df <- plot_df
        df$Cluster <- as.factor(df$Cluster)

        cluster_levels_vec <- levels(df$Cluster)
        n_clusters <- length(cluster_levels_vec)

        colors <- c(
          "#7A6FF0","#FF7FB0","#FFAA66","#A78BFA","#FF9EC7",
          "#8FD8C8","#4DB6AC","#FFD54F","#BA68C8","#64B5F6",
          "#E57373","#AED581","#FFB74D","#4DD0E1","#F06292","#7986CB"
        )

        centers <- df %>%
          group_by(Cluster) %>%
          summarise(
            UMAP1 = mean(UMAP1),
            UMAP2 = mean(UMAP2),
            .groups = 'drop'
          )

        p <- ggplot()

        # ===============================
        # 展示风格：density(密度云) / scatter(纯散点) / contour(轮廓)
        # ===============================
        style <- "{style}"
        if (style == "density") {{
          for(i in 1:n_clusters){{
            current_level <- cluster_levels_vec[i]
            cluster_data <- subset(df, Cluster == current_level)
            p <- p +
              stat_density_2d(
                data = cluster_data,
                aes(UMAP1, UMAP2),
                geom = "polygon",
                fill = colors[i],
                alpha = 0.05,
                color = NA,
                contour = TRUE,
                bins = 2
              )
          }}
        }} else if (style == "contour") {{
          for(i in 1:n_clusters){{
            current_level <- cluster_levels_vec[i]
            cluster_data <- subset(df, Cluster == current_level)
            p <- p +
              stat_density_2d(
                data = cluster_data,
                aes(UMAP1, UMAP2),
                geom = "polygon",
                fill = colors[i],
                alpha = 0.12,
                color = colors[i],
                contour = TRUE,
                bins = 5
              )
          }}
        }}
        # style == "scatter" 时不绘制密度/轮廓，仅保留散点与标签

        # ===============================
        # label
        # ===============================
        x_min <- min(df$UMAP1)
        x_max <- max(df$UMAP1)
        y_min <- min(df$UMAP2)
        y_max <- max(df$UMAP2)

        left_x <- x_min + (x_max - x_min) * 0.02
        top_y  <- y_max - (y_max - y_min) * 0.02

        clusters <- cluster_levels_vec

        label_df <- data.frame(
          x = left_x,
          y = top_y - (seq_along(clusters) - 1) * (y_max - y_min) * 0.025,
          label = paste0(
            clusters, " = ",
            unlist(topic_names[as.character(clusters)])
          )
        )

        p <- p +

          geom_point(
            data = df,
            aes(UMAP1, UMAP2, color = Cluster),
            size = 1.6,
            alpha = 0.85
          ) +

          geom_text(
            data = centers,
            aes(UMAP1, UMAP2, label = Cluster),
            color = "black",
            size = 3.5,
            fontface = "bold"
          ) +

          geom_text(
            data = label_df,
            aes(x = x, y = y, label = label),
            inherit.aes = FALSE,
            hjust = 0,
            size = 2.2,
            fontface = "bold",
            family = "Times"
          ) +

          scale_color_manual(values = colors) +

          theme_void(base_family = "Times") +
          theme(
            legend.position = "none"
          )

        ggsave("{image_path}", p, width = 7, height = 7, dpi = 500)

        print(paste("Saved: {image_path}"))
        """

        robjects.r(r_code)

        print(f"✅ UMAP 聚类图已生成：{image_path}")
        return image_path

# ================================
# 入口函数
# ================================
def generate_umap_cluster_img(file_path, save_filename="umap_cluster_R.png",
                               text_column=None, cluster_column=None, theme_column=None,
                               save_dir=None, style="density"):
    """
    生成 UMAP 聚类图的便捷函数

    Parameters:
    -----------
    file_path : str
        Excel 或 CSV 文件路径
    save_filename : str, optional
        保存的文件名，默认 "umap_cluster_R.png"
    text_column : str, optional
        文本列名，为 None 则自动识别
    cluster_column : str, optional
        聚类标签列名，为 None 则自动识别
    theme_column : str, optional
        主题名称列名（如 "en_theme"），用于在图中显示人类可读的类别名

    Returns:
    --------
    str
        生成的图片路径
    """
    visualizer = UMAPClusterVisualizer(file_path)
    visualizer.load_data(
        text_column=text_column,
        cluster_column=cluster_column,
        theme_column=theme_column
    )
    visualizer.generate_embeddings()
    visualizer.reduce_dimensions()
    return visualizer.plot_with_r(save_filename=save_filename, save_dir=save_dir, style=style)


# ================================
# CLI
# ================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成 UMAP 聚类可视化图")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 Excel 文件路径")
    parser.add_argument("--output", "-o", type=str, default="umap_cluster_R.png", help="输出文件名")

    args = parser.parse_args()

    generate_umap_cluster_img(
        file_path=args.input,
        save_filename=args.output
    )
