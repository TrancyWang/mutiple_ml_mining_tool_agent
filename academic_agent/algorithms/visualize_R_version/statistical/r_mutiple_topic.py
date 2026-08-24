# =========================
# main.py （最终稳定版：排序 + TopN + facet 稳定 + 自动清理）
# =========================

import pandas as pd
import subprocess
import re
import os
import glob
from typing import Optional


def generate_topic_visualization(input_file_path: str, output_dir: Optional[str] = None,
                                 cluster_column: str = "cluster",
                                 keyword_column: str = "keywords"):
    """
    生成多主题关键词可视化图表
    
    Args:
        input_file_path: 输入 Excel 文件路径（包含 cluster 和 keywords 列）
        output_dir: 输出目录路径，默认为 data_analysis_result/text_mining_imgs
        cluster_column: 聚类列名，默认 "cluster"，支持从文件字段中自动选择
        keyword_column: 关键词列名，默认 "keywords"，支持从文件字段中自动选择
    
    Returns:
        str: 生成的图片文件路径
    """
    # ====== 0. 参数验证和初始化 ======
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"输入文件不存在：{input_file_path}")
    
    # 如果没有指定输出目录，使用默认路径
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_dir, 'data_analysis_result', 'text_mining_imgs')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入文件名（不含扩展名），用于生成输出文件名
    input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    output_plot_path = os.path.join(output_dir, f"{input_filename}_keywords_plot.png")
    temp_csv_path = os.path.join(output_dir, f"{input_filename}_keywords_data.csv")
    temp_r_script_path = os.path.join(output_dir, f"{input_filename}_plot_cluster.R")
    
    print(f"📝 输入文件：{input_file_path}")
    print(f"💾 输出目录：{output_dir}")
    print(f"🖼️  输出图片：{output_plot_path}")
    
    # ====== 1. 读取 Excel（支持有表头/无表头两种格式）======
    print("\n📖 步骤 1: 读取 Excel 文件...")
    df_raw = pd.read_excel(input_file_path)
    print(f"📋 文件实际列名：{list(df_raw.columns)}")

    # ⭐ 格式修复：智能匹配列名（忽略大小写/空格）
    def _smart_match(columns, target):
        target_norm = target.strip().lower().replace(" ", "")
        for c in columns:
            if c.strip().lower().replace(" ", "") == target_norm:
                return c
        return None

    actual_cluster = _smart_match(df_raw.columns, cluster_column)
    actual_keyword = _smart_match(df_raw.columns, keyword_column)

    if actual_cluster is None or actual_keyword is None:
        # 给出清晰的错误，而不是静默回退导致后续 R 崩溃
        missing = []
        if actual_cluster is None:
            missing.append(f"'{cluster_column}'")
        if actual_keyword is None:
            missing.append(f"'{keyword_column}'")
        raise ValueError(
            f"文件不包含指定列 {', '.join(missing)}。"
            f"当前文件可用列为：{list(df_raw.columns)}。"
            f"\n💡 请在下拉框中选择实际存在的列（例如聚类列通常为 'cluster'，"
            f"关键词列通常为 'keywords'）。"
        )

    # 重命名到标准列名，避免列名带特殊字符传给 R 出问题
    df_raw = df_raw.rename(columns={
        actual_cluster: "cluster_internal",
        actual_keyword: "keyword_internal"
    })
    cluster_column = "cluster_internal"
    keyword_column = "keyword_internal"
    print(f"✅ 已映射到列：聚类='{actual_cluster}', 关键词='{actual_keyword}'")
    
    # 丢弃完全为空的行
    df_raw = df_raw.dropna(subset=[cluster_column, keyword_column], how="all")
    print(f"✅ 成功读取 {len(df_raw)} 行数据")
    
    
    # ====== 2. 解析关键词（带格式修复，支持多种写法）======
    def parse_keywords(s):
        """
        解析关键词字段，支持多种常见格式并自动修复：
        1. 标准元组字符串:  [('词', 0.5), ('词2', 0.3)]
        2. 单引号/双引号混合: [("词", 0.5)]
        3. 用冒号分隔:       词:0.5; 词2:0.3  /  词: 0.5
        4. 用短横线分隔:     词 - 0.5
        5. 词(0.5) 形式:     词(0.5)
        6. 纯文本词（无数值）: 按出现顺序赋默认权重 1.0
        返回: [(term, float_value), ...]
        """
        text = str(s).strip()
        if not text or text.lower() in ("nan", "none", "[]"):
            return []
        
        pairs = []
        
        # 格式 1/2: ('term', value) 或 ("term", value)，允许内部含空格/标点
        tuples = re.findall(r"[\(\[]\s*['\"](.+?)['\"]\s*,\s*([0-9]+\.?[0-9]*)\s*[\)\]]", text)
        if tuples:
            for term, val in tuples:
                term = term.strip()
                if term:
                    pairs.append((term, float(val)))
            return pairs
        
        # 格式 3: term:value 或 term: value（分号/逗号/空格分隔）
        colon = re.findall(r"([^;,:]+?)\s*:\s*([0-9]+\.?[0-9]*)", text)
        if colon:
            for term, val in colon:
                term = term.strip()
                if term:
                    pairs.append((term, float(val)))
            return pairs
        
        # 格式 4: term - value
        dash = re.findall(r"([^-]+?)\s*-\s*([0-9]+\.?[0-9]*)", text)
        if dash:
            for term, val in dash:
                term = term.strip()
                if term:
                    pairs.append((term, float(val)))
            return pairs
        
        # 格式 5: term(value)
        paren = re.findall(r"([^(]+?)\s*\(([0-9]+\.?[0-9]*)\)", text)
        if paren:
            for term, val in paren:
                term = term.strip()
                if term:
                    pairs.append((term, float(val)))
            return pairs
        
        # 格式 6: 纯文本词（被分隔符分开），赋默认权重
        tokens = re.split(r"[;,\n|]+", text)
        for t in tokens:
            t = t.strip().strip("'\"").strip()
            if t and not re.fullmatch(r"[0-9]+\.?[0-9]*", t):
                pairs.append((t, 1.0))
        return pairs

    def normalize_keywords(raw_pairs):
        """
        ⭐ 将解析出的关键词统一转换为标准的「元组词典格式」：
            [('词', 权重), ('词2', 权重2), ...]
        - 词去重（同名词只保留权重最大的一条）
        - 权重统一转为 float
        - 按权重降序排列
        - 非法/空词剔除
        返回规整后的 list[tuple(str, float)]
        """
        norm = {}
        for term, val in raw_pairs:
            term = str(term).strip()
            if not term:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 1.0
            # 同名词保留最高权重
            if term not in norm or val > norm[term]:
                norm[term] = val
        # 转成标准元组列表并按权重降序
        std = [(term, norm[term]) for term in norm]
        std.sort(key=lambda x: x[1], reverse=True)
        return std

    def to_standard_tuple_str(std_pairs):
        """把标准元组列表序列化为 '[('词', 值), ...]' 字符串（便于存档/调试）"""
        return "[" + ", ".join(
            f"('{term}', {val:.4g})" for term, val in std_pairs
        ) + "]"
    
    
    # ====== 3. 转换数据（统一转为标准元组词典格式）======
    print("\n📊 步骤 2: 解析和转换数据...")
    data = []
    standardized_records = []  # 规整后的 (cluster, 标准元组字符串) 备查
    
    for _, row in df_raw.iterrows():
        cluster = f"{str(row[cluster_column])}"
        # 3.1 先解析（兼容各种杂乱写法）
        raw_pairs = parse_keywords(row[keyword_column])
        # 3.2 统一归一化为标准元组词典格式
        std_pairs = normalize_keywords(raw_pairs)
        standardized_records.append((cluster, to_standard_tuple_str(std_pairs)))
        
        # 3.3 基于标准格式展开成 (cluster, term, value) 供 R 使用
        for term, value in std_pairs:
            data.append([cluster, term, value])
    
    df = pd.DataFrame(data, columns=["cluster", "term", "value"])
    print(f"✅ 解析到 {len(df)} 个关键词（已统一为标准元组词典格式）")
    
    # （可选）把规整后的标准格式写回一个 CSV，方便用户核对
    try:
        std_df = pd.DataFrame(standardized_records, columns=["cluster", "keywords_standard"])
        std_csv = os.path.join(os.path.dirname(temp_csv_path),
                               f"{input_filename}_keywords_standard.csv")
        std_df.to_csv(std_csv, index=False, encoding="utf-8-sig")
        print(f"📝 已导出标准化关键词至：{std_csv}")
    except Exception as e:
        print(f"⚠️ 标准化关键词存档失败（不影响主流程）：{e}")
    
    # ⭐ 空数据保护：避免 R 端 facet 报错
    if len(df) == 0:
        raise ValueError(
            f"未能从 '{actual_keyword}' 列解析出任何关键词。\n"
            f"请确认该列包含形如 [('词', 0.5), ('词2', 0.3)] 的关键词数据，"
            f"或逗号/冒号分隔的词:权重文本。\n"
            f"当前该列前 3 个样本为：{df_raw[keyword_column].head(3).tolist()}"
        )
    
    
    # ====== 4. Top N（工业级稳定写法）======
    TOP_N = 10
    
    # ⭐ 全局排序（第二层保险）
    df = df.sort_values(["cluster", "value"], ascending=[True, False])
    
    df_processed = (
        df.groupby("cluster", group_keys=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    
    print(f"✅ 每个主题保留 Top {TOP_N} 个关键词，共 {len(df_processed)} 条记录")
    
    
    # ====== 5. 输出 CSV ======
    print(f"\n💾 步骤 3: 保存临时 CSV 文件...")
    df_processed.to_csv(temp_csv_path, index=False)
    print(f"✅ 临时 CSV 已保存：{temp_csv_path}")
    
    
    # ====== 6. R 脚本（彻底修复排序 + facet 问题）======
    print(f"\n🎨 步骤 4: 生成 R 可视化脚本...")
    r_code = '''
library(ggplot2)
library(dplyr)

df <- read.csv("''' + temp_csv_path.replace(chr(92), '/') + '''", stringsAsFactors = FALSE)

# ⭐ 手动实现“分组排序 key”
df <- df %>%
  group_by(cluster) %>%
  arrange(desc(value), .by_group = TRUE) %>%
  mutate(term2 = paste0(term, "_", cluster)) %>%
  ungroup()

df$cluster <- factor(df$cluster, levels = unique(df$cluster))

p <- ggplot(df, aes(x = value, y = reorder(term2, value), fill = cluster)) +
  geom_bar(stat = "identity", width = 0.7) +
  
  facet_wrap(~cluster, scales = "free", ncol = 4) +
  
  scale_y_discrete(labels = function(x) sub("_.+$", "", x)) +
  
  guides(fill = "none") +
  
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(size = 13, face = "bold"),
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 9),
    panel.grid.major.y = element_blank()
  ) +
  
  labs(x = NULL, y = NULL)

ggsave("''' + output_plot_path.replace(chr(92), '/') + '''", p, width = 14, height = 10, dpi = 300)

print(p)
'''
    
    # 写入 R 文件
    with open(temp_r_script_path, "w", encoding="utf-8") as f:
        f.write(r_code)
    print(f"✅ R 脚本已保存：{temp_r_script_path}")
    
    
    # ====== 7. 调用 R ======
    print(f"\n🚀 步骤 5: 执行 R 脚本进行可视化...")
    try:
        result = subprocess.run(
            ["Rscript", temp_r_script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ R 脚本执行成功")
        if result.stdout:
            print(f"R 输出：{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ R 脚本执行失败：{e}")
        if e.stderr:
            print(f"错误信息：{e.stderr}")
        raise
    
    
    # ====== 8. 清理临时文件 ======
    print(f"\n🧹 步骤 6: 清理临时文件...")
    cleanup_temp_files(temp_r_script_path, temp_csv_path)
    
    print(f"\n✅ 可视化完成！图片已保存到：{output_plot_path}")
    return output_plot_path


def cleanup_temp_files(r_script_path: str, csv_path: str):
    """
    清理临时文件
    
    Args:
        r_script_path: R 脚本文件路径
        csv_path: CSV 文件路径
    """
    files_to_clean = [r_script_path, csv_path]
    
    for file in files_to_clean:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ 已删除：{file}")
            except Exception as e:
                print(f"⚠️  删除失败：{file}, 原因：{e}")


# =========================
# 主程序入口（保持向后兼容）
# =========================
if __name__ == "__main__":
    # 默认使用原来的文件路径（向后兼容）
    default_input = "cluster_rd_keywords.xlsx"
    
    if os.path.exists(default_input):
        generate_topic_visualization(default_input)
    else:
        print(f"⚠️  默认输入文件不存在：{default_input}")
        print("请使用函数调用方式并指定输入文件路径")
