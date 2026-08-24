from academic_agent.algorithms.visualization.statistical.statistical_cluster_analyzer import ClusterAnalyzer
from academic_agent.algorithms.visualization.statistical.statistical_cluster_visualizer import ClusterVisualizer
from academic_agent.algorithms.visualization.statistical.statistical_cluster_stats_saver import ClusterStatsSaver


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
    
    # 使用 ClusterAnalyzer 处理数据
    analyzer = ClusterAnalyzer(file_path, sheet_name, text_column, label_column)
    if not analyzer.load_and_process_data():
        return None
    
    # 排序
    analyzer.sort_clusters(sort_by_count)
    
    # 获取统计信息
    cluster_stats = analyzer.get_statistics()
    
    # 打印统计信息
    analyzer.print_statistics()
    
    # 可视化部分
    if show_bar_chart or show_pie_chart:
        visualizer = ClusterVisualizer(analyzer.clusters, analyzer.counts)
        
        # 显示柱状图
        if show_bar_chart:
            visualizer.create_bar_chart()
            
        # 显示饼图
        if show_pie_chart:
            visualizer.create_pie_chart()
    
    # 返回统计结果
    return cluster_stats


def save_cluster_stats(cluster_stats, output_file):
    """
    保存聚类统计结果到文件

    参数:
    cluster_stats (dict): 聚类统计结果字典
    output_file (str): 输出文件路径
    """
    saver = ClusterStatsSaver()
    saver.save_stats(cluster_stats, output_file)


# 使用示例
if __name__ == "__main__":
    result = calculate_cluster(
        file_path="../../../data/c_data/f_llm_cluster_rd.xlsx",
        sheet_name=0,
        text_column=0,
        label_column=1,
        show_bar_chart=True,
        show_pie_chart=False,
        sort_by_count=True
    )

    # 保存统计结果
    if result:
        save_cluster_stats(result, "../../cluster_statistics.csv")

    # 显示情感雷达图
    # sentiment_rida()