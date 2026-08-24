# =========================
# FULL PIPELINE：
# clusters_with_time.xlsx → A.xlsx → topics_over_time → 可视化
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime

# ====== 1. 读取原始数据 ======
df_raw = pd.read_excel("test_clusters.xlsx")

# ====== 2. 添加时间戳（如果没有Timestamp列）======
if "Timestamp" not in df_raw.columns:
    years = list(range(2015, 2024))
    df_raw["Timestamp"] = [datetime(random.choice(years), 1, 1) for _ in range(len(df_raw))]
# ====== 3. 保存为 A.xlsx（保留）======
df_raw.to_excel("A.xlsx", index=False)
# ====== 4. 构建 topics_over_time ======
df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"])

topics_over_time = (
    df_raw.groupby(["cluster", "Timestamp"])
    .size()
    .reset_index(name="Frequency")
    .rename(columns={"cluster": "Topic"})
    .sort_values("Timestamp")
)

# ====== 5. 进入可视化流程 ======
df = topics_over_time.copy()

# ====== 6. 选 Top N topics ======
top_n = 5
top_topics = (
    df.groupby("Topic")["Frequency"]
    .sum()
    .sort_values(ascending=False)
    .head(top_n)
    .index
)

df = df[df["Topic"].isin(top_topics)]

# ====== 7. 归一化 ======
df["Normalized_Freq"] = df.groupby("Timestamp")["Frequency"].transform(
    lambda x: x / x.sum()
)

# ====== 8. 平滑 ======
df["Smooth_Freq"] = df.groupby("Topic")["Normalized_Freq"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# ====== 9. 颜色 ======
palette = sns.color_palette("Set2", n_colors=top_n)

# ====== 10. Topic标签 ======
topic_labels = {
    t: f"Topic {t}" for t in df["Topic"].unique()
}

# ====== 11. 画图 ======
plt.figure(figsize=(11, 6))

for i, topic in enumerate(df["Topic"].unique()):
    subset = df[df["Topic"] == topic]

    plt.plot(
        subset["Timestamp"],
        subset["Smooth_Freq"],
        color=palette[i],
        linewidth=2,
        marker="o",
        markersize=4,
        label=topic_labels.get(topic, f"Topic {topic}")
    )

# ====== 12. 美化 ======
plt.title("Topics over Time", fontsize=16, weight="bold")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Normalized Frequency", fontsize=12)

plt.grid(alpha=0.3)

plt.legend(
    title="Global Topic Representation",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()

# ====== 13. 保存 ======
plt.savefig("topics_over_time_final.png", dpi=300, bbox_inches="tight")

plt.show()