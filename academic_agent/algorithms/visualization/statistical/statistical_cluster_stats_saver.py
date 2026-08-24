import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusterStatsSaver:
    """
    聚类统计结果保存器类
    """

    @staticmethod
    def save_stats(cluster_stats, output_file):
        """
        保存聚类统计结果到文件

        参数:
        cluster_stats (dict): 聚类统计结果字典
        output_file (str): 输出文件路径
        """
        if cluster_stats is None:
            logger.warning("没有统计数据可保存")
            return

        # 保存为CSV格式
        if output_file.endswith('.csv'):
            df = pd.DataFrame.from_dict(cluster_stats, orient='index', columns=['count'])
            df.index.name = 'cluster'
            df.to_csv(output_file)

        # 保存为TXT格式
        elif output_file.endswith('.txt'):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Cluster Distribution:\n")
                f.write("-" * 30 + "\n")
                for cluster, count in cluster_stats.items():
                    f.write(f"{cluster}: {count}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total texts: {sum(cluster_stats.values())}\n")
                f.write(f"Total clusters: {len(cluster_stats)}\n")

        logger.info(f"统计数据已保存到 {output_file}")