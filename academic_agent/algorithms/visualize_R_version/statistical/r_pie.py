# -*- coding: utf-8 -*-
"""
R 版本饼状图生成器（Cluster Distribution Pie Chart）
功能：生成聚类分布饼状图，支持百分比显示和自定义颜色方案
保存到 data_analysis_result/text_mining_imgs 目录
"""

import os
import sys
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class PieChartGenerator:
    """饼状图生成器"""
    
    def __init__(self, data_file_path, save_dir=None):
        self.data_file_path = data_file_path
        
        # 设置图片保存目录
        if save_dir:
            self.save_dir = save_dir
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self.save_dir = os.path.join(project_root, 'data_analysis_result', 'text_mining_imgs')
        
        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.df = None
        self.plot_df = None
    
    def load_data(self, cluster_column="cluster", theme_column=None):
        """
        加载数据并统计聚类分布
        
        Parameters:
        -----------
        cluster_column : str, optional
            聚类标签列名，默认 "cluster"
        theme_column : str, optional
            主题名称列名（如 "en_theme"），用于在饼图中显示人类可读的类别名
        """
        print(f"📂 正在读取数据文件：{self.data_file_path}")
        
        # 读取 Excel 或 CSV
        if self.data_file_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_file_path)
        else:
            self.df = pd.read_excel(self.data_file_path)
        
        print(f"📋 数据形状：{self.df.shape}")
        print(f"📋 列名：{self.df.columns.tolist()}")
        
        # 自动识别 cluster 列
        if cluster_column not in self.df.columns:
            candidates = ["cluster", "Cluster", "topic", "label", "group"]
            for col in candidates:
                if col in self.df.columns:
                    cluster_column = col
                    print(f"🔍 自动识别到 cluster 列：{cluster_column}")
                    break
            else:
                cluster_column = self.df.columns[-1]
                print(f"⚠️ 未找到标准 cluster 列，使用最后一列：{cluster_column}")
        
        # 构建主题名称映射
        self.theme_map = {}
        if theme_column and theme_column in self.df.columns:
            for cid in self.df[cluster_column].unique():
                mask = self.df[cluster_column] == cid
                theme_vals = self.df.loc[mask, theme_column].dropna()
                if len(theme_vals) > 0:
                    self.theme_map[cid] = str(theme_vals.iloc[0])
            print(f"📝 已从列 '{theme_column}' 加载主题名称映射: {self.theme_map}")
        
        # 统计聚类分布
        cluster_counts = self.df[cluster_column].value_counts().sort_index()
        print(f"📊 聚类分布：{cluster_counts.to_dict()}")
        
        # 准备绘图数据
        cluster_ids = cluster_counts.index.tolist()
        self.plot_df = pd.DataFrame({
            "cluster": cluster_ids,
            "count": cluster_counts.values.tolist()
        })
        
        # 标签优先使用主题名称
        self.plot_df["cluster_label"] = [
            self.theme_map.get(cid, f"C{cid}") for cid in cluster_ids
        ]
        
        print(f"✅ 数据加载完成，共 {len(self.plot_df)} 个聚类")
        return self
    
    def generate_pie_chart(self, color_scheme="Paired", show_percentage=True, 
                          save_filename="cluster_distribution_pie_R.png"):
        """
        生成饼状图
        
        Parameters:
        -----------
        color_scheme : str, optional
            颜色方案，默认 "Paleted" (支持最多 12 个类别)
            可选：Set1 (最多 9), Set2 (最多 8), Set3 (最多 12), Paired (最多 12)
        show_percentage : bool, optional
            是否显示百分比，默认 True
        save_filename : str, optional
            保存的文件名，默认 "cluster_distribution_pie_R.png"
        
        Returns:
        --------
        str
            生成的图片路径
        """
        if self.plot_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        print(f"\n🎨 开始生成饼状图...")
        print(f"   颜色方案：{color_scheme}")
        print(f"   显示百分比：{show_percentage}")
        print(f"   类别数量：{len(self.plot_df)}")
        
        # 转换为 R 数据框
        with localconverter(robjects.default_converter + pandas2ri.converter):
            robjects.globalenv["df"] = pandas2ri.py2rpy(self.plot_df)
        
        robjects.globalenv["color_scheme"] = color_scheme
        robjects.globalenv["show_percentage"] = show_percentage
        
        # 计算图片保存路径
        image_path = os.path.join(self.save_dir, save_filename)
        
        # R 绘图代码
        pct_code = "scales::percent(df$count/max(df$count))" if show_percentage else "df$count"
        r_code = f'''
library(ggplot2)
library(RColorBrewer)

# 计算比例和累积值
df$fraction <- df$count / sum(df$count)
df$ymax <- cumsum(df$fraction)
df$ymin <- c(0, head(df$ymax, n=-1))
df$labelPosition <- (df$ymax + df$ymin) / 2
df$label <- paste0(df$cluster_label, "\\n", {pct_code})

# 生成颜色 - 使用 Paired 调色板（支持最多 12 个类别）
n_clusters <- length(unique(df$cluster))
colors <- colorRampPalette(brewer.pal(12, "{color_scheme}"))(n_clusters)

# 绘制饼状图
p <- ggplot(df, aes(ymax=ymax, ymin=ymin, xmax=4, xmin=3, fill=factor(cluster))) +
  geom_rect() +
  coord_polar(theta="y") +
  xlim(c(2, 4)) +
  scale_fill_manual(values = colors) +
  theme_void() +
  theme(legend.position="right") +
  ggtitle("Cluster Distribution")

# 保存图片
ggsave("{image_path}", plot = p, width = 8, height = 6, dpi = 600)

print(paste("✅ 饼状图已保存：{image_path}"))
print(paste("类别数量:", n_clusters))
print(paste("颜色数量:", length(colors)))
'''
        
        print("🔄 正在调用 R 绘图...")
        robjects.r(r_code)
        
        print(f"✅ 饼状图已生成：{image_path}")
        return image_path


# ======================================================
# 便捷函数
# ======================================================
def generate_pie_chart(file_path, cluster_column="cluster", 
                      color_scheme="Set1", show_percentage=True,
                      save_filename="cluster_distribution_pie_R.png",
                      theme_column=None, save_dir=None):
    """
    生成饼状图的便捷函数
    
    Parameters:
    -----------
    file_path : str
        Excel 或 CSV 文件路径
    cluster_column : str, optional
        聚类标签列名，默认 "cluster"
    color_scheme : str, optional
        颜色方案，默认 "Set1"
    show_percentage : bool, optional
        是否显示百分比，默认 True
    save_filename : str, optional
        保存的文件名，默认 "cluster_distribution_pie_R.png"
    theme_column : str, optional
        主题名称列名（如 "en_theme"），用于在饼图中显示人类可读的类别名
    
    Returns:
    --------
    str
        生成的图片路径
    """
    generator = PieChartGenerator(file_path, save_dir=save_dir)
    generator.load_data(cluster_column=cluster_column, theme_column=theme_column)
    image_path = generator.generate_pie_chart(
        color_scheme=color_scheme,
        show_percentage=show_percentage,
        save_filename=save_filename
    )
    return image_path


# ======================================================
# CLI 入口
# ======================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成 R 版本聚类分布饼状图")
    parser.add_argument("--input", "-i", type=str, required=True, 
                       help="输入 Excel 或 CSV 文件路径")
    parser.add_argument("--cluster-column", "-c", type=str, default="cluster",
                       help="聚类标签列名，默认 cluster")
    parser.add_argument("--color-scheme", type=str, default="Set1",
                       help="颜色方案，默认 Set1")
    parser.add_argument("--no-percentage", action="store_true",
                       help="不显示百分比")
    parser.add_argument("--output", "-o", type=str, 
                       default="cluster_distribution_pie_R.png",
                       help="输出文件名")
    
    args = parser.parse_args()
    
    generate_pie_chart(
        file_path=args.input,
        cluster_column=args.cluster_column,
        color_scheme=args.color_scheme,
        show_percentage=not args.no_percentage,
        save_filename=args.output
    )
