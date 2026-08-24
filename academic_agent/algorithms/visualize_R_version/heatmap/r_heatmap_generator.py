import pandas as pd
import numpy as np
import logging
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RHeatmapGenerator:
    """
    R 语言热力图生成器类（修复版）
    """

    def __init__(self):
        """
        初始化 R 环境
        """
        try:
            self.ggplot2 = importr('ggplot2')
            self.grid = importr('grid')  # 保留即可（不再用于布局）
            logger.info("R 包加载成功")
        except Exception as e:
            logger.error(f"R 包加载失败：{e}")
            raise e

    def generate_heatmap(
            self,
            similarity_matrix,
            cluster_labels,
            cluster_ids,
            theme_lines=None,
            save_path="./cluster_similarity_heatmap_R.png"
    ):
        try:
            # =========================
            # 1. 构建 DataFrame
            # =========================
            matrix_df = pd.DataFrame(
                similarity_matrix,
                index=cluster_ids,
                columns=cluster_ids
            )

            # ⭐ 坐标使用纯 cluster 显示名（与柱状图一致：R1 / c1 ...）
            label_mapping = {
                cid: label for cid, label in zip(cluster_ids, cluster_labels)
            }

            matrix_df_melted = matrix_df.reset_index().melt(
                id_vars=['index'],
                var_name='Cluster_X',
                value_name='Similarity'
            )
            matrix_df_melted.columns = ['Cluster_Y', 'Cluster_X', 'Similarity']

            matrix_df_melted['Label_X'] = matrix_df_melted['Cluster_X'].map(label_mapping)
            matrix_df_melted['Label_Y'] = matrix_df_melted['Cluster_Y'].map(label_mapping)

            # =========================
            # 2. 传入 R
            # =========================
            with localconverter(ro.default_converter + pandas2ri.converter):
                ro.globalenv['data'] = ro.conversion.py2rpy(matrix_df_melted)

            # 构建底部主题名说明（en_theme 写在最下面，使用英文标题避免中文）
            if theme_lines:
                grouped_themes = []
                for i in range(0, len(theme_lines), 2):
                    group = theme_lines[i:i+2]
                    grouped_themes.append(" | ".join(group))
                caption_text = "Cluster - Theme Mapping:\\n" + "\\n".join(grouped_themes)
                num_lines = len(grouped_themes) + 1
            else:
                caption_text = ("Heatmap of cosine similarity between clusters. "
                                "Dark blue = higher similarity.")
                num_lines = 1

            bottom_margin = 35 + num_lines * 18

            # =========================
            # 3. R 绘图
            # =========================
            caption_escaped = caption_text.replace('"', '\\\\"')
            r_code = f"""
            library(ggplot2)

            theme_set(theme_minimal(base_size = 14))

            p <- ggplot(data, aes(x = Label_X, y = Label_Y, fill = Similarity)) +

              geom_tile(color = "#e5e5e5", linewidth = 0.25) +

              scale_fill_gradientn(
                colors = c("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
                values = seq(0, 1, length.out = 5),
                limits = c(0, 1),
                name = "Cosine Similarity"
              ) +

              scale_x_discrete(position = "top") +

              labs(
                x = NULL,
                y = NULL,
                title = "Cluster Similarity Heatmap",
                caption = "{caption_escaped}"
              ) +

              coord_fixed() +

              theme(
                axis.text.x = element_text(
                  angle = 45,
                  hjust = 1,
                  size = 11
                ),
                axis.text.y = element_text(size = 11),
                axis.ticks = element_blank(),
                panel.grid = element_blank(),
                legend.position = "right",
                legend.title = element_text(size = 12, face = "bold"),
                legend.text = element_text(size = 10),
                plot.title = element_text(
                  hjust = 0.5,
                  size = 15,
                  face = "bold"
                ),
                plot.caption = element_text(
                  hjust = 0.5,
                  size = 11,
                  face = "bold",
                  margin = margin(t = 15)
                ),
                plot.margin = margin(20, 20, {bottom_margin}, 20)
              )

            ggsave("{save_path}", plot = p, width = 11, height = 9, dpi = 400)

            cat("图已保存: {save_path}\\n")
            """

            ro.r(r_code)

            logger.info(f"R 热力图已生成：{save_path}")
            return save_path

        except Exception as e:
            logger.error(f"生成失败：{e}")
            raise e