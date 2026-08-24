import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取 Excel
df = pd.read_excel("semantic_superclusters.xlsx")

# 2. 按超级簇加权求和（Topic Number 是权重）
weighted = df.groupby("Supercluster")["Topic Number"].sum()

# 3. 绘制图形
plt.figure(figsize=(7, 5))
plt.bar(weighted.index.astype(str), weighted.values)

plt.xlabel("Supercluster")
plt.ylabel("Total Topic Count")
# plt.title("Weighted Distribution of Superclusters")

# 论文风格需要紧凑排版
plt.tight_layout()
plt.savefig("supercluster_weighted_distribution.png", dpi=300)
plt.show()
