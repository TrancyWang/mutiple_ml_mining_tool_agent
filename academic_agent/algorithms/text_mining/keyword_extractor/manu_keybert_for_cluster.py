import pandas as pd
from collections import defaultdict
#from textKeyBert_cluster import  TextKeyBert
from textKeyBert_cluster_sentence_fuse_pro import  TextKeyBert
def load_text_by_label(csv_path, has_header=False):
    """
    读取csv文件，第一列是文本，第二列是类别
    返回: {类别: [文本列表]} 的字典
    """
    if has_header:
        df = pd.read_csv(csv_path,sep='\t', header=0)
        # 假设第一列是文本，第二列是类别
        text_col, label_col = df.columns[0], df.columns[1]
    else:
        df = pd.read_csv(csv_path, sep='\t', header=None)
        df.columns = ["unique_id","text", "cluster"]
        text_col, label_col = "text", "cluster"

    data_dict = defaultdict(list)
    for _, row in df.iterrows():
        data_dict[row[label_col]].append(row[text_col])

    return dict(data_dict)


def excel_to_tsv(excel_path, tsv_path, sheet_name=0):
    """
    读取Excel文件并保存为以 \t 分隔的文件 (TSV)
    :param excel_path: 输入Excel文件路径
    :param tsv_path: 输出TSV文件路径
    :param sheet_name: 需要读取的工作表，默认第一个sheet，可以用名字或索引
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8-sig")  # 使用制表符分隔
    print(f"已保存为 {tsv_path} (制表符分隔)")



# 使用示例
if __name__ == "__main__":
    excel_to_tsv("../data/chinese_version/robo-advisor_remark_cluster_modify_6_cluster_cn.xlsx", "../data/chinese_version/robo-advisor_remark_cluster_modify_6_cluster_cn.csv", sheet_name="Sheet1")
    
    data_dict = load_text_by_label("../data/chinese_version/robo-advisor_remark_cluster_modify_6_cluster_cn.csv", has_header=True)
    kbert_model = TextKeyBert()
    cluster_keywords_dict = {}
    for k, v in data_dict.items():
        #words = kbert_model.get_key_words(v, "lda", res_top_n=50)
        words = kbert_model.get_key_words(v, res_top_n=50)
        cluster_keywords_dict[k] = str(words)

    df_cluster_keywords = pd.DataFrame(list(cluster_keywords_dict.items()), columns=["cluster", "keywords"])
    df_cluster_keywords.to_excel("../data/chinese_version/robo-advisor_remark_keywords_modify_6_cluster_cn_pro.xlsx", sheet_name="sheet1", index=False, header=True)

