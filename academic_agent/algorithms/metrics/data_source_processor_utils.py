import pandas as pd
from collections import defaultdict
from typing import Dict, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_label_text_dict(file_path: str,
                            text_col_index: int = 0,
                            label_col_index: int = 2) -> Dict[str, List[str]]:
    """
    从Excel文件中提取按标签分类的文本字典。

    参数:
        file_path (str): Excel 文件路径
        text_col_index (int): 文本列的索引（从0开始），默认是第1列
        label_col_index (int): 标签列的索引（从0开始），默认是第3列

    返回:
        dict: 形如 {'positive': ['很好', '非常好'], 'neutral': [...]} 的字典
    """
    df = pd.read_excel(file_path)
    text_col = df.iloc[:, text_col_index]
    label_col = df.iloc[:, label_col_index]

    label_dict = defaultdict(list)

    for text, label in zip(text_col, label_col):
        label_dict[str(label)].append(str(text))  # 保证是字符串

    return dict(label_dict)


def  statis_result(path='../data/version1/robo-advisor_remark_cluster_11_cluster.xlsx'):
    df = pd.read_excel(path)  # 默认header=0（第一行为列名）

    # 检查列名是否正确（可选，但建议添加）
    # 如果列名不是"文本"和"类别"，需要替换成实际的列名（例如：df.columns = ["文本", "类别"]）
    # 示例：如果原列名是"内容"和"分类"，则改为：df.columns = ["文本", "类别"]
    # 统计每个类别的条数
    category_counts = df.groupby('cluster').size().reset_index(name='数量')
    # 计算总数
    total = len(df)
    # 计算每个类别占总数的百分比（保留2位小数，带%符号）
    category_counts['占比'] = (category_counts['数量'] / total * 100).round(2).astype(str) + '%'
    # 打印结果
    logger.info("各类别统计结果：")
    logger.info(f"\n{category_counts}")
    # 将结果保存到新的Excel文件
    category_counts.to_excel('类别统计结果.xlsx', index=False)
    logger.info("统计结果已保存到'类别统计结果.xlsx'")


# # ✅ 调用示例
# if __name__ == "__main__":
#     file = '../data/robo-advisor_remark_predicted_sentiment_modify.xlsx'
#     label_text_dict = extract_label_text_dict(file)
#
#     # 打印前几个条目看看结果
#     for label, texts in label_text_dict.items():
#         print(f"{label}: {texts[:3]}")  # 只展示每类前3条文本
