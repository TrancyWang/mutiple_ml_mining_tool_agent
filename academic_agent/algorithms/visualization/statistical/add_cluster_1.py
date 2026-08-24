import pandas as pd
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 文件路径（修改为实际路径）
input_file = "../../../data/c_data/f_llm_cluster_rd.xlsx"
output_file = "../../../data/c_data/f_llm_cluster_rd.xlsx"

# 检查输入文件是否存在
if not os.path.exists(input_file):
    raise FileNotFoundError(f"输入文件不存在: {input_file}")

try:
    # 读取数据
    df = pd.read_excel(input_file, sheet_name=0)

    # 检查列数
    if len(df.columns) < 2:
        raise ValueError("Excel文件需要至少两列数据")

    # 转换第二列为数值
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

    # 检查是否有无效值
    if df.iloc[:, 1].isna().any():
        raise ValueError("第二列包含非数值数据")

    # 执行计算
    df.iloc[:, 1] = df.iloc[:, 1] + 1

    # 保存结果
    df.to_excel(output_file, index=False)
    logger.info(f"处理完成！结果已保存到 {output_file}")

except Exception as e:
    logger.error(f"处理失败: {str(e)}")
