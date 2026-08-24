# -*- coding: utf-8 -*-
"""
review_semantic_splitter_final.py

功能：
1. 原始评论（只有一列文本） → 语义单元
2. 句子 / 子句级切割
3. 相邻补充型语义单元轻度合并
4. 自动生成 remark_id
5. 合理文本清洗（保留中文语义标点）
6. 输出 CSV（可直接用于聚类 / 情感分析）

适用场景：
- 酒店 / 民宿 / 平台评论分析
- 情感分析
- 语义聚类
- 用户感知结构建模
"""

import re
import csv
import uuid
from typing import List, Dict
import pandas as pd

# =====================================================
# 1. 语义切割：评论 → 句子 / 子句级语义单元
# =====================================================

def split_review_to_semantic_units(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []

    text = str(text).strip()

    # Step 1：句子级切分（中英文）
    sentences = re.split(r'(?<=[。！？.!?])', text)
    units = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Step 2：子句级切分（逗号 / 分号）
        clauses = re.split(r'(?<=[,，;；])', sent)

        for clause in clauses:
            clause = clause.strip()

            if len(clause) < 2:
                continue

            # 过滤纯符号
            if re.fullmatch(r'[^\w\u4e00-\u9fa5]+', clause):
                continue

            units.append(clause)

    return units


# =====================================================
# 2. 相邻语义单元轻度合并（核心）
# =====================================================

MERGE_PREFIXES = (
    "让", "使", "所以", "因此",
    "也", "还", "而且",
    "没有", "不", "很", "非常",
    "挺", "特别", "比较"
)

SUMMARY_PHRASES = (
    "总体", "整体", "总的来说",
    "非常满意", "强烈推荐",
    "值得推荐", "下次还会",
    "必住", "首选"
)

def is_summary_sentence(text: str) -> bool:
    return any(p in text for p in SUMMARY_PHRASES)

def can_merge(prev: str, curr: str) -> bool:
    """
    判断 curr 是否是 prev 的补充 / 结果说明
    """
    # curr 太长 → 不合
    if len(curr) > 8:
        return False

    # prev 是总结句 → 不合
    if is_summary_sentence(prev):
        return False

    # curr 以补充型词语开头 → 可合
    if curr.startswith(MERGE_PREFIXES):
        return True

    return False

def merge_adjacent_units(units: List[str]) -> List[str]:
    if not units:
        return []

    merged = []
    buffer = units[0]

    for curr in units[1:]:
        if can_merge(buffer, curr):
            buffer = buffer + "，" + curr
        else:
            merged.append(buffer)
            buffer = curr

    merged.append(buffer)
    return merged


# =====================================================
# 3. 批量处理评论（自动生成 remark_id）
# =====================================================

def process_reviews(
    reviews: List[str],
    use_uuid: bool = False
) -> List[Dict]:

    records = []

    for idx, review in enumerate(reviews):
        remark_id = (
            f"remark_{uuid.uuid4().hex[:8]}"
            if use_uuid else
            f"remark_{idx + 1}"
        )

        # 切割
        units = split_review_to_semantic_units(review)

        # ✅ 轻度合并
        units = merge_adjacent_units(units)

        for u_idx, unit in enumerate(units):
            records.append({
                "remark_id": remark_id,
                "unit_id": f"{remark_id}_u{u_idx + 1}",
                "semantic_unit": unit
            })

    return records


# =====================================================
# 4. 文本清洗（⚠️ 不破坏语义）
# =====================================================

def clean_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""

    text = str(text).strip()

    # 去除各种引号（单引号、双引号、中文引号）
    text = re.sub(r'[\'\'\']', '', text)
    text = re.sub(r'["""\']', '', text)

    # 合并多余空白
    text = re.sub(r'\s+', '', text)

    # ⚠️ 保留中文标点（，。！？）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text)

    return text


def clean_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df[text_column] = df[text_column].apply(clean_text)
    df = df[df[text_column].str.len() > 1]
    return df


def remove_empty_rows_from_csv(file_path: str, output_path: str = None, text_column: str = "semantic_unit"):
    """
    从CSV文件中删除空行
    
    Args:
        file_path: 输入CSV文件路径
        output_path: 输出CSV文件路径，如果为None则覆盖原文件
        text_column: 文本列名，默认为"semantic_unit"
    
    Returns:
        DataFrame: 删除空行后的DataFrame
    """
    df = pd.read_csv(file_path)
    
    # 删除指定列为空值的行
    df = df.dropna(subset=[text_column])
    
    # 删除指定列为空字符串的行
    df = df[df[text_column].str.strip() != '']
    
    # 如果没有指定输出路径，则覆盖原文件
    if output_path is None:
        output_path = file_path
    
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已删除空行并保存到：{output_path}（共 {len(df)} 行）")
    
    return df


# =====================================================
# 5. CSV 保存
# =====================================================

def save_to_csv(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{output_path}（共 {len(df)} 条）")


# =====================================================
# 6. 使用示例（你只需要改这里的路径）
# =====================================================

if __name__ == "__main__":

    # # === ① 读取原始数据（只有一列文本） ===
    # input_file = "../../data/hotel/journal5 raw data.xlsx"
    # text_column = "contents"
    #
    # raw_df = pd.read_excel(input_file)
    # reviews = raw_df[text_column].dropna().tolist()
    #
    # print(f"原始评论数：{len(reviews)}")
    #
    # # === ② 语义切割 + 合并 ===
    # records = process_reviews(reviews, use_uuid=False)
    # units_df = pd.DataFrame(records)
    #
    # print(f"生成语义单元数（合并前清洗）：{len(units_df)}")
    #
    # # === ③ 清洗文本 ===
    # units_df = clean_dataframe(units_df, "semantic_unit")
    #
    # print(f"清洗后语义单元数：{len(units_df)}")
    # df = units_df[["semantic_unit"]]
    # df.to_csv("semantic_units_text_only.csv", index=False, encoding="utf-8-sig")
    # print("完成，仅保留 semantic_unit 列")

    units_df = remove_empty_rows_from_csv("../../data/hotel/semantic_units_text_only.csv", "../../data/hotel/semantic_units_text_only.csv" )
    # === ⑤ 示例查看 ===
    print("\n示例语义单元：")
    print(units_df.head(10))
