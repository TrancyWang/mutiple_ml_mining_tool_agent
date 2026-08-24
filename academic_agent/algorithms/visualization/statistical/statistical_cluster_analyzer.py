import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    聚类数据分析器类
    """

    def __init__(self, file_path, sheet_name=0, text_column=0, label_column=1):
        """
        初始化聚类分析器

        参数:
        file_path (str): Excel 文件路径
        sheet_name (str/int): 工作表名称或索引，默认为第一个工作表
        text_column (int): 文本内容所在的列索引，默认为第0列
        label_column (int): cluster label 所在的列索引，默认为第1列
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.text_column = text_column
        self.label_column = label_column
        self.df = None
        self.cluster_counts = None
        self.clusters = None
        self.counts = None

    def load_and_process_data(self):
        """
        加载并处理聚类数据

        返回:
        bool: 数据加载和处理是否成功
        """
        # 检查文件路径
        if self.file_path is None:
            raise ValueError("请提供 Excel 文件路径")

        # 读取 Excel 文件
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=0)

            # 提取 cluster labels
            cluster_labels = self.df.iloc[:, self.label_column].tolist()

            # 统计每个 cluster 的数量
            self.cluster_counts = Counter(cluster_labels)

            # 提取 cluster 名称和对应数量
            self.clusters = list(self.cluster_counts.keys())
            self.counts = list(self.cluster_counts.values())
            
            return True
            
        except FileNotFoundError:
            logger.error(f"文件 {self.file_path} 未找到，请检查文件路径是否正确")
            return False
        except Exception as e:
            logger.error(f"处理文件时发生错误: {e}")
            return False

    def sort_clusters(self, by_count=False):
        """
        对聚类结果进行排序

        参数:
        by_count (bool): 是否按数量排序，默认为 False（按标签名称排序）
        """
        if self.clusters is None or self.counts is None:
            raise ValueError("尚未加载数据，请先调用 load_and_process_data 方法")

        # 排序
        if by_count:
            # 按数量降序排序
            sorted_pairs = sorted(zip(self.clusters, self.counts), key=lambda x: x[1], reverse=True)
        else:
            # 按 cluster 名称排序
            sorted_pairs = sorted(zip(self.clusters, self.counts), key=lambda x: str(x[0]))

        self.clusters, self.counts = zip(*sorted_pairs)

    def get_statistics(self):
        """
        获取聚类统计信息

        返回:
        dict: 包含聚类统计信息的字典
        """
        if self.clusters is None or self.counts is None:
            raise ValueError("尚未加载数据，请先调用 load_and_process_data 方法")
        
        stats = dict(zip(self.clusters, self.counts))
        return stats

    def print_statistics(self):
        """
        打印聚类统计信息
        """
        if self.clusters is None or self.counts is None:
            raise ValueError("尚未加载数据，请先调用 load_and_process_data 方法")
            
        logger.info("Cluster Distribution:")
        logger.info("-" * 30)
        for cluster, count in zip(self.clusters, self.counts):
            logger.info(f"{cluster}: {count}")
        logger.info("-" * 30)
        logger.info(f"Total texts: {sum(self.counts)}")
        logger.info(f"Total clusters: {len(self.clusters)}")