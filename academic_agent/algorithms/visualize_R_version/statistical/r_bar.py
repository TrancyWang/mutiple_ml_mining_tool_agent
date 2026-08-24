# -*- coding: utf-8 -*-
"""
R 版本柱状图生成器
提供可复用的柱状图生成函数
"""
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import os
import re


def _build_theme_map(df, cluster_column, theme_column):
    """
    构建 cluster_id → theme_name 的映射字典
    
    Parameters:
    -----------
    df : DataFrame
        数据
    cluster_column : str
        聚类标签列名
    theme_column : str or None
        主题名称列名（如 "en_theme"），为 None 则不构建
    
    Returns:
    --------
    dict: {cluster_id: theme_name}
    """
    theme_map = {}
    if theme_column and theme_column in df.columns:
        cluster_ids = df[cluster_column].unique()
        for cid in cluster_ids:
            mask = df[cluster_column] == cid
            theme_vals = df.loc[mask, theme_column].dropna()
            if len(theme_vals) > 0:
                theme_map[cid] = str(theme_vals.iloc[0])
        print(f"📝 已从列 '{theme_column}' 加载主题名称映射: {theme_map}")
    return theme_map


def generate_cluster_bar_chart(file_path, save_dir=None, color="#4C72B0", 
                                sort_order="descending", show_caption=True,
                                cluster_column="cluster", theme_column=None):
    """
    生成 R 版本聚类分布柱状图
    
    Parameters:
    -----------
    file_path : str
        Excel 或 CSV 文件路径，必须包含指定的聚类列
    save_dir : str, optional
        保存目录，默认为 None（使用当前目录）
    color : str, optional
        柱子颜色，默认 "#4C72B0"（蓝色）
    sort_order : str, optional
        排序方式："descending"(降序), "ascending"(升序), "none"(不排序)
    show_caption : bool, optional
        是否显示详细说明
    cluster_column : str, optional
        聚类列的列名，默认 "cluster"，支持字符串类型的聚类标签（如 "R0", "R1" 等）
    theme_column : str, optional
        主题名称列名（如 "en_theme"），用于在 x 轴和 caption 中显示人类可读的类别名
    
    Returns:
    --------
    str
        生成的图片路径
    """
    
    # 1. 读取数据
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            print(f"✅ 已读取聚类数据，共 {len(df)} 条记录")
        except Exception as e:
            print(f"❌ 读取 sheet1 失败：{str(e)}")
            raise
    
    # 2. 构建主题映射（优先从 theme_column，兼容旧的 sheet2 映射表）
    topic_dict = _build_theme_map(df, cluster_column, theme_column)
    
    # 如果未指定 theme_column，尝试兼容旧的 sheet2 映射表
    if not theme_column:
        try:
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            if len(sheet_names) > 1:
                for sheet_name in sheet_names[1:]:
                    try:
                        topic_mapping = pd.read_excel(file_path, sheet_name=sheet_name)
                        if 'cluster' in topic_mapping.columns:
                            for _, row in topic_mapping.iterrows():
                                cluster_key = str(row['cluster'])
                                if cluster_key not in topic_dict:
                                    topic_name = str(row.get('topic', f'Cluster {cluster_key}'))
                                    topic_dict[cluster_key] = topic_name
                            print(f"✅ 已从 sheet '{sheet_name}' 读取 cluster-topic 映射表")
                            break
                    except Exception:
                        continue
        except Exception:
            pass
    
    # 3. 统计聚类分布
    cluster_col = cluster_column if cluster_column in df.columns else df.columns[-1]
    print(f"📊 使用聚类列：'{cluster_col}'")
    cluster_counts = df[cluster_col].value_counts().sort_index()
    
    # 4. 准备绘图数据
    plot_data = {
        "cluster": cluster_counts.index.tolist(),
        "count": cluster_counts.values.tolist()
    }
    plot_df = pd.DataFrame(plot_data)
    
    # 5. 排序处理
    if sort_order == "descending":
        plot_df = plot_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    elif sort_order == "ascending":
        plot_df = plot_df.sort_values(by="count", ascending=True).reset_index(drop=True)
    
    # 6. 生成 x 轴下方的显示标签（只显示 cluster 本身）
    #    规则：字母+数字（如 R1）直接使用；纯数字（如 1,2,3）加前缀 c -> c1,c2
    def format_cluster_display(cid):
        str_cid = str(cid)
        # 纯数字（允许负号/小数）则加前缀 c
        if re.fullmatch(r'-?\d+(\.\d+)?', str_cid):
            return f"c{str_cid}"
        # 字母+数字等，直接使用
        return str_cid

    plot_df["display_label"] = [format_cluster_display(cid) for cid in plot_df["cluster"]]
    
    # cluster_label 用于 caption 说明（优先主题名，其次 display_label）
    plot_df["cluster_label"] = [
        topic_dict.get(cid, topic_dict.get(str(cid), plot_df.loc[idx, "display_label"]))
        for idx, cid in enumerate(plot_df["cluster"])
    ]
    
    # 7. 生成说明文字
    if show_caption:
        labels = []
        for _, row in plot_df.iterrows():
            cluster_id = row['cluster']
            count = int(row['count'])
            str_cid = str(cluster_id)

            if str_cid in topic_dict or cluster_id in topic_dict:
                topic_name = topic_dict.get(cluster_id, topic_dict.get(str_cid, ""))
                # 主题名含非 ASCII（如中文）则不写入图片，避免中文
                if topic_name and all(ord(ch) < 128 for ch in str(topic_name)):
                    label = f"{row['display_label']}={topic_name}"
                else:
                    label = f"{row['display_label']}"
            else:
                label = f"{row['display_label']}"

            labels.append(label)
        
        grouped_labels = []
        for i in range(0, len(labels), 2):
            group = labels[i:i+2]
            grouped_labels.append(" | ".join(group))
        
        note_text = "\n".join(grouped_labels)
        num_lines = len(grouped_labels)
        bottom_margin = 15 + num_lines * 15
    else:
        note_text = ""
        bottom_margin = 10
    
    # 7. 设置保存路径
    if save_dir is None:
        # 计算图片保存目录：data_analysis_result/text_mining_imgs
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 向上追溯 3 层到项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        save_dir = os.path.join(project_root, 'data_analysis_result', 'text_mining_imgs')
        
        # 添加调试信息
        print(f"\n📁 当前文件路径：{current_dir}")
        print(f"📁 项目根目录：{project_root}")
        print(f"📁 保存目录：{save_dir}")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cluster_distribution_R.png")
    
    print(f"\n💾 图片保存路径：{save_path}")
    print(f"📂 保存目录：{save_dir}")
    print(f"✅ 目录存在：{os.path.exists(save_dir)}\n")
    
    # 8. 转换为 R 数据框
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            rdf = ro.conversion.py2rpy(plot_df)
        
        ro.globalenv["df"] = rdf
        ro.globalenv["note_text"] = note_text
        ro.globalenv["bar_color"] = color
        ro.globalenv["show_caption"] = show_caption
        ro.globalenv["bottom_margin"] = bottom_margin
        
        # 9. R 绘图代码（关键修复：移除 print(p)，只保存不显示）
        r_code = '''
        # 设置为非交互式图形后端，防止 SIGILL 错误
        Sys.setenv(RSTUDIO_ACTIVE = "0")
        Sys.setenv(R_BROWSER = "false")
        Sys.setenv(R_PDFVIEWER = "false")
        
        library(ggplot2)
        
        # 使用 Cairo 图形设备（更稳定）
        tryCatch({
            library(Cairo)
            # 打开 Cairo PNG 设备
            CairoPNG(
                filename = "%s",
                width = 10,
                height = 7,
                units = "in",
                dpi = 600
            )
        }, error = function(e) {
            # 如果 Cairo 不可用，使用默认的 png 设备
            png(
                filename = "%s",
                width = 10,
                height = 7,
                units = "in",
                res = 600
            )
        })
        
        df$display_label <- factor(df$display_label, levels = df$display_label)
        
        p <- ggplot(df, aes(x = display_label, y = count)) +
          
          geom_bar(stat = "identity", fill = bar_color, width = 0.6) +
          
          geom_text(
            aes(label = count),
            vjust = -0.6,
            size = 3.2
          ) +
          
          labs(
            x = "Cluster",
            y = "Number of Texts",
            caption = if(show_caption) note_text else ""
          ) +
          
          theme_classic() +
          
          theme(
            axis.text.x = element_text(size = 11),
            axis.title = element_text(size = 12),
            
            panel.border = element_rect(
              color = "black",
              fill = NA,
              size = 0.4
            ),
            
            axis.line = element_line(size = 0.3),
            
            plot.caption = element_text(
              hjust = 0.5,
              size = 8,
              lineheight = 1.5,
              margin = margin(t = 10, r = 20, b = 10, l = 20)
            ),
            plot.margin = margin(10, 10, bottom_margin, 10)
          )
        
        # 打印到打开的设备
        print(p)
        
        # 关闭图形设备
        dev.off()
        ''' % (save_path.replace('\\', '/'), save_path.replace('\\', '/'))
        
        # 执行 R 代码并捕获输出
        result = ro.r(r_code)
        
        print(f"✅ R 柱状图已生成：{save_path}")
        return save_path
        
    except SystemExit as e:
        # 处理 Python 退出的情况
        print(f"❌ 检测到系统退出请求：{e.code}")
        print("💡 这可能是因为图形窗口被关闭导致的")
        raise
        
    except Exception as e:
        print(f"❌ 生成 R 柱状图失败：{str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 确保清理 R 环境（使用正确的 rpy2 语法）
        try:
            # 方式 1: 使用 R 命令清理特定变量
            ro.r('rm(df, note_text, bar_color, show_caption)')
            # 方式 2: 垃圾回收
            ro.r('gc()')
        except Exception as cleanup_error:
            # 清理失败不影响主流程
            pass


# ======================
# 测试代码（仅在直接运行时执行）
# ======================
if __name__ == "__main__":
    # 示例：从 Excel 文件生成柱状图
    test_file = "../../../data_analysis_result/clusters.xlsx"
    if os.path.exists(test_file):
        result_path = generate_cluster_bar_chart(
            file_path=test_file,
            save_dir="./output",
            color="#4C72B0",
            sort_order="descending"
        )
        print(f"测试完成：{result_path}")
    else:
        print(f"⚠️ 测试文件不存在：{test_file}")