# ============================================================
# 决策树 —— 信息增益（熵）· 不划分训练/测试
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score

# ★ Mac 中文字体（Windows 用户改成 ['SimHei', 'Microsoft YaHei']）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ========== 只需改这两行 ==========
DATA = "renew.csv"       # 你的数据文件
DEPTH = 3                # 树最大深度
TARGET = None            # None = 自动用最后一列
# ==================================

# ---------- 读数据 ----------
df = pd.read_csv(DATA)

if TARGET is None:
    TARGET = df.columns[-1]
    print(f"[自动识别] 目标列 = '{TARGET}'")

X_df = pd.get_dummies(df.drop(columns=[TARGET]))
X = X_df.values
y = df[TARGET].values
names = list(X_df.columns)


# ---------- 熵 ----------
def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.unique(y, return_counts=True)[1] / len(y)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


# ---------- 手动算根节点最优分裂（用全量数据） ----------
print("=" * 60)
print("根节点计算过程（信息增益，全量数据）")
print("=" * 60)

parent = entropy(y)
print(f"父节点熵 = -Σp·log₂p = {parent:.4f}")

best = None
for j in range(X.shape[1]):
    for t in np.unique(X[:, j]):
        mask = X[:, j] <= t
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        w = mask.mean() * entropy(y[mask]) + (~mask).mean() * entropy(y[~mask])
        gain = parent - w
        if best is None or gain > best[0]:
            best = (gain, j, t, w)

gain, j, t, w = best
print(f"最优分裂：{names[j]} <= {t}")
print(f"  加权熵   = {w:.4f}")
print(f"  信息增益 = {gain:.4f}")


# ---------- 训练（全部数据） ----------
model = DecisionTreeClassifier(criterion="entropy", max_depth=DEPTH, random_state=42)
model.fit(X, y)

print(f"\n全量数据准确率：{accuracy_score(y, model.predict(X)):.4f}")
print("\n最终树结构：")
print(export_text(model, feature_names=names))


# ---------- 画图 ----------
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=names, class_names=list(model.classes_),
          filled=True, rounded=True, precision=2)
plt.title("决策树 - 信息增益（熵，全量数据）")
plt.savefig("tree_entropy.png", dpi=120, bbox_inches="tight")
print("\n图已保存：tree_entropy.png")
plt.show()
