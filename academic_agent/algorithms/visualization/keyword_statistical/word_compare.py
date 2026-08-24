# sentiment_wordshift_excel_jieba_butterfly_refactored.py
import pandas as pd
import shifterator as sh
import matplotlib.pyplot as plt
import numpy as np
from academic_agent.algorithms.visualization.keyword_statistical.doc_utils import safe_fix_string, fix_to_dict_auto, looks_like_pairs

# ===== 中文字体设置 =====
plt.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False
# ===== Step 1: 读取 Excel =====
def load_excel(path):
    return pd.read_excel(path)


# ===== Step 2: 合并人工词典与频率 =====
def merge_freqs(freq_excel, freq_dict):
    """
    freq_excel: dict，已有词频或情绪分
    freq_dict: dict，人工词典及权重
    返回合并后的字典
    """
    merged = freq_excel.copy()
    for word, val in freq_dict.items():
        merged[word] = merged.get(word, 0) + val
    return merged


# ===== Step 3: Shifterator 计算词移分数 =====
def compute_shift_scores(freqs1, freqs2):
    """
    使用Shifterator计算EntropyShift
    返回词 -> shift score字典
    """
    shift = sh.EntropyShift(type2freq_1=freqs1, type2freq_2=freqs2, base=2)
    return shift.get_shift_scores()


# ===== Step 4: 获取TopN正负词 =====
def get_top_words(shift_scores, top_n=20):
    pos = sorted([(w, s) for w, s in shift_scores.items() if s > 0],
                 key=lambda x: x[1], reverse=True)[:top_n]
    neg = sorted([(w, s) for w, s in shift_scores.items() if s < 0],
                 key=lambda x: x[1])[:top_n]

    # 对齐长度
    max_len = max(len(pos), len(neg))
    pos += [("", 0)] * (max_len - len(pos))
    neg += [("", 0)] * (max_len - len(neg))

    # 拆分词和分数，并反转，使最大权重在上方
    pos_words, pos_scores = zip(*pos)
    neg_words, neg_scores = zip(*neg)
    return (list(pos_words)[::-1], list(pos_scores)[::-1],
            list(neg_words)[::-1], list(neg_scores)[::-1])


# ===== Step 5: 绘制分栏词移图 =====
def plot_wordshift(pos_words, pos_scores, neg_words, neg_scores, title="Word Shift", save_path=None):
    max_len = len(pos_words)
    y = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 左侧：正面，绿色
    ax.barh(y, [-s for s in pos_scores], color="green", alpha=0.7)
    for i, (w, s) in enumerate(zip(pos_words, pos_scores)):
        if w:
            ax.text(-s - 0.01, i, f"{max_len - i}. {w}", ha="right", va="center", fontsize=11, color="green")

    # 右侧：负面，红色
    ax.barh(y, [abs(s) for s in neg_scores], color="red", alpha=0.7)
    for i, (w, s) in enumerate(zip(neg_words, neg_scores)):
        if w:
            ax.text(abs(s) + 0.01, i, f"{max_len - i}. {w}", ha="left", va="center", fontsize=11, color="red")

    # 中线 + 网格
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Shift Contribution Score")
    plt.title(title, fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()




def load_sentiment_dicts_from_excel(file_path):
    """
    更鲁棒的从 Excel 中读取情感词典。
    - 优先在表中搜索第一个/第二个“list of (key, value)”样式的单元格（跳过常见表头 like 'keywords'）
    - 若找不到，退回按行读取前两行并尝试自动修复解析
    返回 [negative_dict, positive_dict]
    """
    df = pd.read_excel(file_path, header=None, dtype=str)  # 全作为字符串读取以避免类型问题

    # 常见应该被视为“表头/标签”的词，不当作词典内容
    header_tokens = {"keywords", "keyword", "label", "labels", "keys", "terms"}

    # 1) 在整个表里按行列顺序搜寻候选项，跳过 header_tokens
    candidates = []
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            raw = df.iat[r, c]
            if raw is None:
                continue
            rs = safe_fix_string(raw).strip()
            if not rs:
                continue
            low = rs.lower()
            if low in header_tokens:
                continue
            if looks_like_pairs(rs):
                # debug print
                print(f"[DEBUG] Found candidate at row {r+1}, col {c+1} -> {repr(rs)}")
                candidates.append(rs)
                if len(candidates) >= 2:
                    break
        if len(candidates) >= 2:
            break

    sentiment_dicts = []

    # 如果通过全表搜索到足够的两个候选（优先用它们）
    if len(candidates) >= 2:
        for i, content in enumerate(candidates[:2]):
            try:
                parsed = fix_to_dict_auto(content)
                sentiment_dicts.append(parsed)
            except Exception as e:
                print(f"解析候选第{i+1}项失败，原因：{e}")
                if i == 0:
                    sentiment_dicts.append({"不好": 0.8, "问题": 0.7})
                else:
                    sentiment_dicts.append({"好": 0.8, "优秀": 0.7})
        return sentiment_dicts

    # 2) 如果没有找到足够的候选项，退回按你原先的“前两行”逻辑并尝试解析（更宽容）
    # 读取前两行（若不足行则扩展）
    rows_to_try = min( max(2, df.shape[0]), df.shape[0] )
    for i in range(2):
        # 取第 i 行的第二列（若没有第二列则取第一列），如果超出范围则使用空字符串
        if i < df.shape[0]:
            raw = df.iat[i, 1] if df.shape[1] > 1 else df.iat[i, 0]
        else:
            raw = ""
        print(f"[DEBUG] Row {i+1} Raw -> {repr(raw)}")
        try:
            if raw is None or safe_fix_string(raw).strip().lower() in header_tokens:
                raise ValueError("单元格是空或表头标签")
            fixed = fix_to_dict_auto(raw)
            sentiment_dicts.append(fixed)
        except Exception as e:
            print(f"解析第 {i+1} 行失败，使用默认词典。原因：{e}")
            if i == 0:
                sentiment_dicts.append({"不好": 0.8, "问题": 0.7})
            else:
                sentiment_dicts.append({"好": 0.8, "优秀": 0.7})

    # 确保返回的是两个字典，即使前面步骤失败也要提供默认值
    while len(sentiment_dicts) < 2:
        if len(sentiment_dicts) == 0:
            sentiment_dicts.append({"不好": 0.8, "问题": 0.7})
        else:
            sentiment_dicts.append({"好": 0.8, "优秀": 0.7})
            
    return sentiment_dicts

# ===== Step 6: 主流程 =====
def get_compare_res (input_file, output_img):
    # 人工词典（权重已计算好）
    dict_list = load_sentiment_dicts_from_excel(input_file)
    print(dict_list)
    # 合并频率
    freqs_neg = merge_freqs({}, dict_list[0])
    freqs_pos = merge_freqs({}, dict_list[1])
    # 计算 shift scores
    shift_scores = compute_shift_scores(freqs_neg, freqs_pos)
    # 获取TopN正负词
    pos_words, pos_scores, neg_words, neg_scores = get_top_words(shift_scores, top_n=20)
    # 绘图
    plot_wordshift(pos_words, pos_scores, neg_words, neg_scores,
                   title="Positive vs Negative",
                   save_path=output_img)

if __name__ == "__main__":
    get_compare_res("cluster_keywords.xlsx", "test.jpg")