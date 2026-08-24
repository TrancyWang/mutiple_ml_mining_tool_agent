###数据合并
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataMeger(object):

    def __init__(self):
       pass


    def merge_remark_files(self, file1, file2, file3):


        # 读取第一个 CSV 文件（只保留 content 列）
        df1 = pd.read_csv(file1, usecols=['content'])
        # 读取第二个 CSV 文件（只保留 content 列）
        df2 = pd.read_csv(file2, usecols=['content'])
        # 合并两个 DataFrame（纵向拼接）
        merged_df = pd.concat([df1, df2], ignore_index=True)
        # （可选）保存合并后的结果到新文件
        merged_df.to_csv(file3, index=False)
        # 打印前 5 行查看结果
        logger.info(f"合并结果前5行:\n{merged_df.head()}")

    def merge_remark_files_list (self, file_list, file3):

        lists = []
        # 读取第一个 CSV 文件（只保留 content 列）
        for file in file_list:
            df1 = pd.read_csv(file, usecols=['content'])
            df1 = df1.drop_duplicates(subset=None, keep='first', inplace=False)
            lists.append(df1)
            # 合并两个 DataFrame（纵向拼接）
        merged_df = pd.concat(lists, ignore_index=True)
            # （可选）保存合并后的结果到新文件
        merged_df.to_csv(file3, index=False)
            # 打印前 5 行查看结果
        logger.info(f"合并结果前5行:\n{merged_df.head()}")

    def duplicate_file (self, org_file,target_file):

        # 读取第一个 CSV 文件（只保留 content 列）
            df1 = pd.read_csv(org_file, usecols=['content'])
            df1 = df1.drop_duplicates(subset=None, keep='first', inplace=False)
            df1.to_csv(target_file, index=False)
            # 打印前 5 行查看结果
            logger.info(f"去重结果前5行:\n{df1.head()}")

data_meger = DataMeger()
org_file1="../datasrc/微博原始评论06.21.csv"
# org_file2="../datasrc/雪球原始帖子评论robo-advisor_remark.csv"
# org_file3="../datasrc/微信.csv"

# ./datasrc/merged_robo-advisor_remark_3.csv
target_file3="../data/general_dicts.txt"
target_file4="../datasrc/merged_robo-advisor_remark_3.csv"
#data_meger.merge_remark_files(org_file1,org_file2,target_file3)

file_path_list=  [org_file1 ]

# data_meger.merge_remark_files_list(file_path_list,target_file3)
data_meger.duplicate_file(target_file3, target_file3 )