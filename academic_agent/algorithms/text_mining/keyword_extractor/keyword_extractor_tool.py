# -*- coding: utf-8 -*-
"""
关键词抽取工具 - 支持参数化调用
可用于 GUI 或命令行调用
"""
import os
import sys
import pandas as pd
from collections import defaultdict

# 添加当前目录到 Python 路径
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from textKeyBert_cluster_sentence_fuse_pro import TextKeyBert


def load_text_by_label(csv_path, has_header=False):
    """
    读取 csv 文件，第一列是文本，第二列是类别
    返回：{类别：[文本列表]} 的字典
    """
    if has_header:
        df = pd.read_csv(csv_path, sep='\t', header=0)
        # 假设第一列是文本，第二列是类别
        text_col, label_col = df.columns[0], df.columns[1]
    else:
        df = pd.read_csv(csv_path, sep='\t', header=None)
        df.columns = ["unique_id", "text", "cluster"]
        text_col, label_col = "text", "cluster"

    data_dict = defaultdict(list)
    for _, row in df.iterrows():
        data_dict[row[label_col]].append(row[text_col])

    return dict(data_dict)


def excel_to_tsv(excel_path, tsv_path, sheet_name=0):
    """
    读取 Excel 文件并保存为以 \t 分隔的文件 (TSV)
    :param excel_path: 输入 Excel 文件路径
    :param tsv_path: 输出 TSV 文件路径
    :param sheet_name: 需要读取的工作表，默认第一个 sheet，可以用名字或索引
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"已保存为 {tsv_path} (制表符分隔)")


def extract_keywords(input_file_path, output_file_path=None, res_top_n=50):
    """
    提取关键词的主函数
    
    :param input_file_path: 输入文件路径（Excel 或 CSV）
    :param output_file_path: 输出文件路径（可选，默认为输入文件同目录下的 keywords_results.xlsx）
    :param res_top_n: 每个类别提取的关键词数量，默认 50
    :return: 关键词 DataFrame
    """
    print(f"\n{'='*60}")
    print("🔑 开始关键词提取...")
    print(f"{'='*60}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"输入文件不存在：{input_file_path}")
    
    # 确定文件类型
    file_ext = os.path.splitext(input_file_path)[1].lower()
    
    # 如果是 CSV 文件，先转换为 Excel
    if file_ext == '.csv':
        print(f"📄 检测到 CSV 文件，正在转换格式...")
        temp_tsv = input_file_path.replace('.csv', '_temp.tsv')
        excel_to_tsv(input_file_path, temp_tsv)
        input_file_path = temp_tsv
        file_ext = '.tsv'
    
    # 如果是 TSV 文件，直接读取
    if file_ext in ['.tsv', '.csv']:
        print(f"📖 正在读取 TSV 文件...")
        data_dict = load_text_by_label(input_file_path, has_header=True)
    else:
        # Excel 文件
        print(f"📖 正在读取 Excel 文件...")
        df = pd.read_excel(input_file_path)
        
        # 假设第一列是文本，第二列是类别
        text_col, label_col = df.columns[0], df.columns[1]
        
        data_dict = defaultdict(list)
        for _, row in df.iterrows():
            data_dict[row[label_col]].append(row[text_col])
        
        data_dict = dict(data_dict)
    
    print(f"✅ 成功读取 {sum(len(v) for v in data_dict.values())} 条数据")
    print(f"📂 共 {len(data_dict)} 个类别")
    
    # 初始化 KeyBERT 模型
    print(f"\n⚙️  正在加载 KeyBERT 模型...")
    kbert_model = TextKeyBert()
    print(f"✅ 模型加载完成")
    
    # 提取关键词
    print(f"\n🔍 正在提取关键词...")
    cluster_keywords_dict = {}
    for k, v in data_dict.items():
        if len(v) > 0:
            print(f"   处理类别：{k} ({len(v)} 条文本)")
            words = kbert_model.get_key_words(v, res_top_n=res_top_n)
            cluster_keywords_dict[k] = str(words)
    
    # 创建结果 DataFrame
    df_keywords = pd.DataFrame(list(cluster_keywords_dict.items()), columns=["cluster", "keywords"])
    
    # 确定输出文件路径
    if output_file_path is None:
        # 使用默认输出路径（输入文件同目录下）
        input_dir = os.path.dirname(input_file_path)
        input_filename = os.path.basename(input_file_path)
        name_without_ext = os.path.splitext(input_filename)[0]
        output_file_path = os.path.join(input_dir, f"{name_without_ext}_keywords_results.xlsx")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    print(f"\n💾 正在保存结果到：{output_file_path}")
    df_keywords.to_excel(output_file_path, sheet_name="Sheet1", index=False, header=True)
    print(f"✅ 关键词提取完成！")
    print(f"{'='*60}\n")
    
    return df_keywords


if __name__ == "__main__":
    # ===== 示例用法 =====
    
    # 示例 1: 使用默认输出路径
    # input_file = "../data/chinese_version/robo-advisor_remark_cluster_modify_6_cluster_cn.xlsx"
    # extract_keywords(input_file)
    
    # 示例 2: 指定输出路径
    # input_file = "../data/chinese_version/robo-advisor_remark_cluster_modify_6_cluster_cn.xlsx"
    # output_file = "../data/results/keywords_results.xlsx"
    # extract_keywords(input_file, output_file, res_top_n=50)
    
    # 示例 3: 从命令行参数读取
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else None
        top_n = int(sys.argv[3]) if len(sys.argv) >= 4 else 50
        
        try:
            extract_keywords(input_file, output_file, top_n)
        except Exception as e:
            print(f"❌ 错误：{str(e)}")
            sys.exit(1)
    else:
        print("使用方法:")
        print("  python keyword_extractor_tool.py <输入文件路径> [输出文件路径] [top_n]")
        print("\n参数说明:")
        print("  输入文件路径  - Excel 或 CSV 文件路径（必需）")
        print("  输出文件路径  - 结果文件路径（可选，默认在输入文件同目录）")
        print("  top_n        - 每个类别提取的关键词数量（可选，默认 50）")
        print("\n示例:")
        print('  python keyword_extractor_tool.py "../data/test.xlsx"')
        print('  python keyword_extractor_tool.py "../data/test.xlsx" "../results/keywords.xlsx" 30')
