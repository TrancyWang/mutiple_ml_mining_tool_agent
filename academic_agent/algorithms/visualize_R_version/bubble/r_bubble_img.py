import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sentence_transformers import SentenceTransformer
import rpy2.robjects as ro


class TopicVisualizer:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        # 使用绝对路径：获取当前文件所在目录，然后向上查找 pretrain_models 目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 当前目录：.../text_mining_tools_client_cpu/src_code/visualize_R_version_stage_5/bubble
        # 需要向上追溯 3 层到项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        self.model_base_dir = os.path.join(project_root, 'pretrain_models')
        self.df = None
        self.sentences = None
        self.cluster_ids = None
        self.cluster_counts = None
        self.cluster_labels = None
        self.embeddings = None
        self.cluster_centers = None
        self.cluster_coords = None
        self.cluster_topic = None   # 🔥 新增

    # ① 加载数据
    def load_data(self, sheet_name=0, content_column="content", cluster_column="cluster"):
        """
        加载数据
        :param sheet_name: Excel 工作表名称或索引，默认为第一个工作表 (0)
        :param content_column: 文本内容列名
        :param cluster_column: 聚类标签列名
        """
        # 处理 sheet_name 为 None 的情况，使用默认值 0（第一个工作表）
        if sheet_name is None:
            sheet_name = 0
        
        self.df = pd.read_excel(self.data_file_path, sheet_name=sheet_name)
        
        # 检查 DataFrame 是否正确加载
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"读取 Excel 失败，期望得到 DataFrame，但得到了 {type(self.df)}")
        
        # 保存列名引用，以便后续使用
        self.content_column = content_column
        self.cluster_column = cluster_column
        
        # 验证指定的列是否存在
        if content_column not in self.df.columns:
            raise ValueError(f"列 '{content_column}' 不存在于数据中，可用列：{list(self.df.columns)}")
        if cluster_column not in self.df.columns:
            raise ValueError(f"列 '{cluster_column}' 不存在于数据中，可用列：{list(self.df.columns)}")

        self.sentences = self.df[content_column].tolist()

        self.cluster_ids = sorted(self.df[cluster_column].unique())
        self.cluster_counts = self.df[cluster_column].value_counts().sort_index().to_dict()

        self.cluster_labels = [
            f"Cluster {cid}"
            for cid in self.cluster_ids
        ]
        
        # 🔥 新增：优先从 cluster_topic 读取，其次从 sheet2 读取 topic 名称映射词典
        if self.cluster_topic is not None and len(self.cluster_topic) > 0:
            # 优先使用 cluster_topic 中的映射
            self.topic_names = self.cluster_topic
            print(f"✅ 从 cluster_topic 加载 {len(self.topic_names)} 个主题名称映射")
        else:
            # 如果 cluster_topic 为空，则从 sheet2 读取
            try:
                topic_df = pd.read_excel(self.data_file_path, sheet_name='sheet2')
                # 假设第一列是 cluster_id，第二列是 topic name
                self.topic_names = dict(zip(topic_df.iloc[:, 0], topic_df.iloc[:, 1]))
                print(f"✅ 从 sheet2 成功加载 {len(self.topic_names)} 个主题名称映射")
            except Exception as e:
                print(f"⚠️ 无法从 sheet2 读取主题名称：{e}")
                print("📝 使用默认命名规则（Cluster + 编号）")
                self.topic_names = {}

    # ② embedding
    def generate_embeddings(self):
        embedding_folder = os.path.join(self.model_base_dir, 'bge-cn')
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = SentenceTransformer(embedding_folder, device=device)

        self.embeddings = model.encode(
            self.sentences,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    # ③ cluster center
    def compute_cluster_centers(self):
        centers = []

        for cid in self.cluster_ids:
            cluster_vecs = self.embeddings[self.df[self.cluster_column] == cid]
            center = cluster_vecs.mean(axis=0)
            centers.append(center)

        self.cluster_centers = np.vstack(centers)

    # 🔥 ④ sentiment（模拟 or 可替换真实情感）
    def compute_sentiment(self):
        # 这里先用随机数（你后面可以换成真实情感模型）
        np.random.seed(42)
        self.cluster_sentiment = {
            cid: np.round(np.random.uniform(0.2, 0.9), 2)
            for cid in self.cluster_ids
        }

    # ⑤ MDS
    def reduce_dimensions(self):
        mds = MDS(n_components=2, random_state=42)
        self.cluster_coords = mds.fit_transform(self.cluster_centers)

    # ⑥ 导出数据 + R 绘图（升级版）
    def plot_with_r(self, save_csv="intertopic_data.csv", save_dir=None):
        coords = self.cluster_coords
        
        total = sum(self.cluster_counts.values())
                
        # 🔥 生成 topic 名称标签
        topic_labels = [
            f"{self.topic_names.get(cid, f'Cluster {cid}')}"
            for cid in self.cluster_ids
        ]
        
        export_df = pd.DataFrame({
            "cluster": self.cluster_ids,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "size": [self.cluster_counts[cid] / total * 100 for cid in self.cluster_ids],  # 🔥 %
            "sentiment": [self.cluster_sentiment[cid] for cid in self.cluster_ids],
            "label": topic_labels
        })
    
        # 计算图片保存目录：优先使用用户指定的 save_dir，否则回退默认目录
        if save_dir and str(save_dir).strip():
            img_save_dir = str(save_dir).strip()
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上追溯 3 层到项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            img_save_dir = os.path.join(project_root, 'data_analysis_result', 'text_mining_imgs')

        # 确保目录存在
        os.makedirs(img_save_dir, exist_ok=True)

        # CSV 与图片都保存到同一目录
        csv_path = os.path.join(img_save_dir, os.path.basename(save_csv))
        export_df.to_csv(csv_path, index=False)
        print(f"Exported: {csv_path}")

        # 计算图片保存路径（使用绝对路径）
        image_path = os.path.join(img_save_dir, "intertopic_map_R.png")
    
        # =========================
        # 🔥 R 代码（升级版）
        # =========================
        r_code = f"""
        # 自动安装（防止报错）
        packages <- c("ggplot2", "ggrepel")
        installed <- packages %in% rownames(installed.packages())
    
        if (any(!installed)) {{
          install.packages(packages[!installed], repos="https://cloud.r-project.org")
        }}
    
        library(ggplot2)
        library(ggrepel)
    
        df <- read.csv("{csv_path}")
    
        p <- ggplot(df, aes(x = x, y = y)) +
    
          geom_point(aes(size = size, color = sentiment),
                     shape = 16,
                     alpha = 0.85) +
    
          geom_text_repel(aes(label = label),
                          size = 4,
                          max.overlaps = 100) +
    
          scale_size(range = c(5, 20), name = "% of corpus", guide = guide_legend(override.aes = list(color = 'lightblue'))) +
    
          scale_color_gradient2(
            low = "#8c2d04",
            mid = "#f7f7f7",
            high = "#2b8cbe",
            midpoint = 0.5,
            name = "Mean positive probability"
          ) +
    
          geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
          geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    
          theme_minimal(base_size = 14) +
    
          theme(
            legend.position = "right",
            plot.title = element_text(face = "bold", hjust = 0.5)
          ) +
    
          ggtitle("Intertopic Distance Map with Cluster")
    
        ggsave("{image_path}", p, width = 10, height = 7, dpi = 300)
    
        print(paste("Saved: {image_path}"))
        """

        ro.r(r_code)


if __name__ == "__main__":
    visualizer = TopicVisualizer(
        data_file_path="clusters.xlsx"
    )

    visualizer.load_data()
    visualizer.generate_embeddings()
    visualizer.compute_cluster_centers()
    visualizer.compute_sentiment()   # 🔥 新增
    visualizer.reduce_dimensions()
    visualizer.plot_with_r()