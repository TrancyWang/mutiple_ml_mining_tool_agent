"""可视化与图表生成工具集 - Qwen-Agent Function Calling Tools

封装词云、聚类可视化、模型评估图表等生成功能。
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import time
import re

# 添加路径
BASE_DIR = Path(__file__).parent.parent
CJK_FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc") if Path(
    "/System/Library/Fonts/STHeiti Medium.ttc"
).exists() else None

# 导入文本挖掘工具以共享数据
from .text_mining_tools import text_mining_tools
from academic_agent.infrastructure.workspace_manager import workspace_manager
from academic_agent.infrastructure.analysis_artifacts import artifact_output_root
from academic_agent.infrastructure.runtime_paths import bundled_resources_root


class VisualizationTools:
    """可视化工具集"""

    def set_data(self, data: Optional[pd.DataFrame]) -> None:
        """兼容统一工具注册表；实际数据源仍由 text_mining_tools 维护。"""
        if data is not None:
            text_mining_tools.current_data = data
    
    @property
    def current_data(self) -> Optional[pd.DataFrame]:
        return text_mining_tools.current_data

    def auto_detect_text_column(self) -> Optional[str]:
        return text_mining_tools.auto_detect_text_column()

    def _get_output_path(self, suffix: str) -> str:
        """生成带时间戳的唯一输出文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        root = artifact_output_root(
            text_mining_tools.current_file_path,
            text_mining_tools.current_source_scope,
        )
        return str(root / f"{suffix}_{timestamp}.png")

    def generate_wordcloud(self,
                          text_column: Optional[str] = None,
                          title: str = "词云图") -> Dict[str, Any]:
        """生成词云图"""
        try:
            if self.current_data is None:
                return {"success": False, "error": "请先加载数据"}
            
            if text_column is None:
                text_column = self.auto_detect_text_column()
                if text_column is None:
                    return {"success": False, "error": "无法自动检测到文本列，请明确指定。"}

            if text_column not in self.current_data.columns:
                return {"success": False, "error": f"列 '{text_column}' 不存在"}
            
            all_text = ' '.join(self.current_data[text_column].dropna().astype(str))
            
            import jieba
            from wordcloud import WordCloud
            
            # 使用更全的停用词表
            stop_words_path = bundled_resources_root() / "data" / "stop_words.txt"
            stop_words = set(open(stop_words_path, encoding='utf-8').read().splitlines()) if stop_words_path.exists() else set()

            words = " ".join([word for word in jieba.cut(all_text) if word not in stop_words and len(word) > 1])
            
            if not words:
                return {"success": False, "error": "没有足够的有效词汇来生成词云图。"}

            wc = WordCloud(font_path='SimHei.ttf', background_color="white", max_words=200, width=800, height=400)
            wc.generate(words)
            
            save_path = self._get_output_path("wordcloud")
            wc.to_file(save_path)
            
            return {
                "success": True,
                "message": "词云图生成完成。",
                "image_path": save_path,
            }
            
        except Exception as e:
            # 检查是否为字体文件错误
            if isinstance(e, OSError) and 'font' in str(e).lower():
                 return {"success": False, "error": "生成词云图失败：缺少SimHei.ttf字体文件。请将字体文件放置在项目根目录下。"}
            return {"success": False, "error": f"词云图生成失败: {str(e)}"}
    
    def plot_distribution(self,
                           column: str,
                           title: Optional[str] = None) -> Dict[str, Any]:
        """绘制数值或类别列的分布图"""
        try:
            if self.current_data is None:
                return {"success": False, "error": "请先加载数据"}
            
            if column not in self.current_data.columns:
                return {"success": False, "error": f"列 '{column}' 不存在"}

            plt.figure(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(self.current_data[column]):
                sns.histplot(self.current_data[column].dropna(), kde=True)
                plot_type = "直方图"
            else:
                sns.countplot(y=self.current_data[column].dropna())
                plot_type = "计数图"

            final_title = title or f"'{column}' 列分布{plot_type}"
            plt.title(final_title, fontsize=14)
            plt.xlabel("值" if pd.api.types.is_numeric_dtype(self.current_data[column]) else "数量")
            plt.ylabel(column)
            plt.tight_layout()

            save_path = self._get_output_path(f"{column}_distribution")
            plt.savefig(save_path, dpi=150)
            plt.close()

            return {
                "success": True,
                "message": f"'{column}' 的分布图已生成。",
                "image_path": save_path,
            }
        except Exception as e:
            return {"success": False, "error": f"分布图生成失败: {str(e)}"}

    def plot_line_chart(self, x_column: str, y_column: str,
                        title: Optional[str] = None) -> Dict[str, Any]:
        """绘制折线图，兼容“2004年1月”“2004-01”等中文月份。"""
        try:
            if self.current_data is None:
                return {"success": False, "error": "请先加载数据"}
            missing = [c for c in (x_column, y_column) if c not in self.current_data.columns]
            if missing:
                return {"success": False, "error": f"列不存在: {missing}"}

            frame = self.current_data[[x_column, y_column]].copy().dropna()
            if frame.empty:
                return {"success": False, "error": "没有可绘制的数据"}
            frame[y_column] = pd.to_numeric(frame[y_column], errors="coerce")
            frame = frame.dropna(subset=[y_column])
            if frame.empty:
                return {"success": False, "error": f"'{y_column}' 不是可绘制的数值列"}

            def normalize_time(value: Any) -> Any:
                text = str(value).strip()
                match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月?", text)
                if match:
                    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)
                parsed = pd.to_datetime(text, errors="coerce")
                return parsed if not pd.isna(parsed) else text

            frame["__x__"] = frame[x_column].map(normalize_time)
            if pd.api.types.is_datetime64_any_dtype(frame["__x__"]):
                frame = frame.sort_values("__x__")
                labels = frame["__x__"].dt.strftime("%Y-%m")
            else:
                labels = frame[x_column].astype(str)

            plt.figure(figsize=(10, 6))
            plt.plot(range(len(frame)), frame[y_column].to_numpy(), marker="o", linewidth=2)
            plt.xticks(range(len(frame)), labels, rotation=45, ha="right")
            plt.title(title or f"{y_column} 随 {x_column} 的变化", fontproperties=CJK_FONT)
            plt.xlabel(x_column, fontproperties=CJK_FONT)
            plt.ylabel(y_column, fontproperties=CJK_FONT)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            save_path = self._get_output_path("line_chart")
            plt.savefig(save_path, dpi=150)
            plt.close()
            return {"success": True, "message": "折线图生成完成。", "image_path": save_path}
        except Exception as exc:
            plt.close("all")
            return {"success": False, "error": f"折线图生成失败: {exc}"}

# 创建全局工具实例
viz_tools = VisualizationTools()
