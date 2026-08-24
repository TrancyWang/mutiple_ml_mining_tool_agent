# -*- coding: utf-8 -*-
import os
import umap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from academic_agent.algorithms.text_mining.embeddings_models.textEmbedding import TextEmbedding
from academic_agent.algorithms.text_mining.keyword_extractor.textKeyBert_sentiment import  TextKeyBert
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Clustering():
    """
     基于sentence_transformer+kemeans，现在使用的这个方法,原理是使用词向量进行聚类分析
    """
    def __init__(self):
        self.text_embbdinger = TextEmbedding()
        self.kbert_model = TextKeyBert()

    def get_text_embedding(self, sentence):
        vector = self.text_embbdinger.get_query_vector(sentence)
        return vector


    def load_content(self, content_file):
        # 尝试多种编码方式读取文件
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']:
            try:
                with open(content_file, 'r', encoding=encoding) as f:
                    res = [i.strip() for i in f.readlines()]
                logger.info(f"成功使用 {encoding} 编码读取文件")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"无法使用任何支持的编码格式读取文件: {content_file}")
        
        return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)), res))  # list, token2index, index2token

    def load_content_pandas(self, content_file):
        """
        从CSV/TSV或Excel文件加载文本内容
        :param content_file: 文件路径（支持.csv、.tsv、.xlsx、.xls）
        :return: 文本列表，文本到索引的映射，索引到文本的映射
        """
        file_ext = os.path.splitext(content_file)[1].lower()
        
        try:
            # 根据文件扩展名选择读取方式
            if file_ext in ['.xlsx', '.xls']:
                # 读取Excel文件
                logger.info(f"检测到Excel文件，正在读取: {content_file}")
                dataframe = pd.read_excel(content_file, sheet_name=0)
                # 如果有多列，使用第一列作为文本内容
                if len(dataframe.columns) > 0:
                    first_col = dataframe.columns[0]
                    res = dataframe[first_col].dropna().astype(str).tolist()
                else:
                    raise ValueError("Excel文件中没有数据列")
            elif file_ext in ['.csv', '.tsv', '.txt']:
                # 读取CSV/TSV文件，尝试多种编码
                logger.info(f"检测到CSV/TSV文件，正在读取: {content_file}")
                for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']:
                    try:
                        dataframe = pd.read_csv(content_file, sep="\t", skiprows=1, names=['content'], encoding=encoding)
                        logger.info(f"成功使用 {encoding} 编码读取文件")
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    raise ValueError(f"无法使用任何支持的编码格式读取文件: {content_file}")
                res = dataframe['content'].dropna().astype(str).tolist()
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .xlsx, .xls, .csv, .tsv, .txt")
            
            logger.info(f"成功读取 {len(res)} 条文本数据")
            return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)), res))
            
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            raise

    def cluster(self, corpus_path, n_clusters=10,is_silhouette=True, is_reduce_dimension=False,n_neighbors=15,n_components=2, max_iter=300, verbose=0):
        """
        KMeans文本聚类
        DBSCAN聚类的用法
        :param corpus_path: 语料路径（每行一篇）,文章id从0开始
        :param n_clusters: ：聚类类别数目
        :param max_iter: 最大迭代次数
        :param verbose: 详细程度
        :return: {cluster_id1:[text_id1, text_id2]}
        """
        corpus, _, index2token = self.load_content_pandas(corpus_path)
        weights = self.get_text_embedding(corpus)

        ## 如果启用降维，UMAP进行降维
        if is_reduce_dimension:

            X_scaled = StandardScaler().fit_transform(weights)
            # 用 UMAP 降到 2D
            reducer = umap.UMAP(n_neighbors= n_neighbors, n_components=n_components, random_state=42)
            weights = reducer.fit_transform(X_scaled)

        silhouette_list = []
        K_range = range(n_clusters, n_clusters+10)

        #自动选择聚类数（Silhouette系数）
        if is_silhouette:
            best_k, best_score = None, -1
            for k in K_range:
                km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42).fit(weights)
                score = silhouette_score(weights, km.labels_)
                silhouette_list.append(silhouette_score(weights, km.labels_))  # Silhouette 系数
                ###选择最好的k点
                if score > best_score:
                    best_k, best_score = k, score
                    n_clusters = best_k
                    logger.info("best_k is " + str(best_k))
            ##KMEANS聚类的用法
        clf = KMeans(
            n_clusters=n_clusters,
            init='k-means++',  # 初始化方法
            n_init='auto',  # 多次不同初始化选择最好结果
            max_iter=max_iter,  # 最大迭代次数
            verbose=verbose,  # 详细程度
            random_state=42
        )
        y = clf.fit_predict(weights)

        result = {}
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [index2token[text_idx]]
            else:
                result[label_idx].append(index2token[text_idx])
        return result


def excel_to_tsv(excel_path, tsv_path, sheet_name=0):
    """
    读取Excel文件并保存为以 \t 分隔的文件 (TSV)
    :param excel_path: 输入Excel文件路径
    :param tsv_path: 输出TSV文件路径
    :param sheet_name: 需要读取的工作表，默认第一个sheet，可以用名字或索引
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8-sig")  # 使用制表符分隔
    logger.info(f"已保存为 {tsv_path} (制表符分隔)")
