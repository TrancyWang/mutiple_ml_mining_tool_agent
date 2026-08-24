import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sentiment_rida():

    # 定义三个评估指标类别
    categories = ['Precision', 'Recall', 'F1-score']

    # 定义每个情感类别的标签，包含样本数量信息
    labels = [
        'Negative (n=829)',    # 负面情感类别，样本数829
        'Neutral (n=152)',     # 中性情感类别，样本数152
        'Positive (n=2176)'    # 正面情感类别，样本数2176
    ]

    # 每个类别对应的三项评估指标值
    data = [
        [0.88, 0.90, 0.89],  # Negative: Precision=0.88, Recall=0.90, F1=0.89
        [0.96, 0.45, 0.61],  # Neutral: Precision=0.96, Recall=0.45, F1=0.61
        [0.95, 0.98, 0.97]   # Positive: Precision=0.95, Recall=0.98, F1=0.97
    ]

    # 创建雷达图的角度坐标，使图形闭合（首尾相连）
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 添加第一个角度以闭合图形
    data = [d + [d[0]] for d in data]  # 每组数据也添加首个点以闭合图形

    # 创建极坐标子图
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # 绘制每个类别的雷达图线条和填充区域
    for d, label in zip(data, labels):
        ax.plot(angles, d, label=label)      # 绘制线条
        ax.fill(angles, d, alpha=0.08)       # 填充区域，透明度为0.08

    # 设置角度网格标签和坐标轴范围
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)  # 设置角度标签为指标名称
    ax.set_ylim(0, 1)  # 设置径向范围从0到1

    # 设置图表标题和图例
    plt.title('Radar Chart of Classification Metrics per label', fontsize=16)  # 图表标题
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)  # 图例位置设置
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 自动调整布局
    plt.show()  # 显示图表


def calculate_cluster(file_path=None, sheet_name=0, text_column=0, label_column=1,
                     show_bar_chart=True, show_pie_chart=True, sort_by_count=False):
    """
    统计 cluster 文件中的数据并绘制统计图

    参数:
    file_path (str): Excel 文件路径
    sheet_name (str/int): 工作表名称或索引，默认为第一个工作表
    text_column (int): 文本内容所在的列索引，默认为第0列
    label_column (int): cluster label 所在的列索引，默认为第1列
    show_bar_chart (bool): 是否显示柱状图，默认为 True
    show_pie_chart (bool): 是否显示饼图，默认为 True
    sort_by_count (bool): 是否按数量排序，默认为 False（按标签名称排序）
    """
    # 检查文件路径
    if file_path is None:
        raise ValueError("请提供 Excel 文件路径")

    # 读取 Excel 文件
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

        # 提取 cluster labels
        cluster_labels = df.iloc[:, label_column].tolist()

        # 统计每个 cluster 的数量
        cluster_counts = Counter(cluster_labels)

        # 提取 cluster 名称和对应数量
        clusters = list(cluster_counts.keys())
        counts = list(cluster_counts.values())

        # 排序
        if sort_by_count:
            # 按数量降序排序
            sorted_pairs = sorted(zip(clusters, counts), key=lambda x: x[1], reverse=True)
        else:
            # 按 cluster 名称排序
            sorted_pairs = sorted(zip(clusters, counts), key=lambda x: str(x[0]))

        clusters, counts = zip(*sorted_pairs)

        # 显示柱状图
        if show_bar_chart:
            # 创建柱状图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(clusters)), counts,
                          color=plt.cm.Set3(np.linspace(0, 1, len(clusters))))

            # 在每个柱子上显示数值
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=10)

            plt.xlabel('Cluster Labels')
            plt.ylabel('Number of Texts')
            plt.title('Distribution of Texts in Clusters')
            plt.xticks(range(len(clusters)), clusters, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("11 Cluster bar statistical.png")
            plt.show()

        # 显示饼图
        if show_pie_chart and len(clusters) <= 15:  # 避免标签过多时饼图难以阅读
            plt.figure(figsize=(8, 8))
            plt.pie(counts, labels=clusters, autopct='%1.1f%%', startangle=90)
            plt.title('Percentage Distribution of Texts in Clusters')
            plt.axis('equal')
            plt.savefig("11 Cluster pie statistical.png")
            plt.show()
        elif show_pie_chart and len(clusters) > 15:
            logger.info("类别过多，跳过饼图显示")

        # 打印统计信息
        logger.info("Cluster Distribution:")
        logger.info("-" * 30)
        for cluster, count in zip(clusters, counts):
            logger.info(f"{cluster}: {count}")
        logger.info("-" * 30)
        logger.info(f"Total texts: {sum(counts)}")
        logger.info(f"Total clusters: {len(clusters)}")

        # 返回统计结果
        return dict(zip(clusters, counts))

    except FileNotFoundError:
        logger.error(f"文件 {file_path} 未找到，请检查文件路径是否正确")
        return None
    except Exception as e:
        logger.error(f"处理文件时发生错误: {e}")
        return None


def save_cluster_stats(cluster_stats, output_file):
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
        import pandas as pd
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

# 使用示例
result = calculate_cluster(
    file_path="../../../../text_mining/c_data/f_llm_cluster_rd.xlsx",
    sheet_name=0,
    text_column=0,
    label_column=1,
    show_bar_chart=True,
    show_pie_chart=False,
    sort_by_count=True
)

# 保存统计结果
save_cluster_stats(result, "../../../../text_clustering/Kmeans/cluster_statistics.csv")


# sentiment_rida()