# Colab 上正確顯示中文標籤的完整程式碼（含多特徵 Logistic Regression）

# 先安裝 Noto CJK 字型（在 Colab 執行一次即可）
import matplotlib.font_manager as fm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# !apt-get update - qq
# !apt-get install - y fonts-noto-cjk - qq


# 設定中文字型（Colab 專用）
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 0. 讀取資料
df = pd.read_csv('/content/student_depression_dataset.csv')

# 1. Depression → 0/1
if df['Depression'].dtype == object:
    df['Depression'] = df['Depression'].map({'No': 0, 'Yes': 1}).astype(int)

# 2. 去重
df = df.drop_duplicates().reset_index(drop=True)

# 3. 填補 Degree 空值為眾數，並去除「其他」
deg_mode = df['Degree'].mode().iloc[0]
df['Degree'] = df['Degree'].fillna(deg_mode).astype(str).str.strip()
df.loc[df['Degree'] == '其他', 'Degree'] = deg_mode

# 4. 合併為四級


def simplify_degree(x):
    x = x.lower()
    if any(k in x for k in ['phd', '博士']):
        return '博士'
    if any(k in x for k in ['master', 'm.', '碩士']):
        return '碩士'
    if any(k in x for k in [
        'bachelor', 'b.', '大學', 'ba', 'bsc', 'bcom', 'be',
        'mba', 'mcom', 'msc', 'bca', 'barch', 'mpharm', 'bpharm'
    ]):
        return '大學'
    return '高中及以下'


df['Degree4'] = df['Degree'].apply(simplify_degree)

# 5. 填補數值欄缺失（中位數）
numeric_cols = ['Age', 'Academic Pressure',
                'Work Pressure', 'CGPA', 'Study Satisfaction']
medians = df[numeric_cols].median()
df[numeric_cols] = df[numeric_cols].fillna(medians)

# 6. Z-score 剔除極端值（|Z|<3）
z = stats.zscore(df[numeric_cols])
df = df[(np.abs(z) < 3).all(axis=1)].reset_index(drop=True)

# 7. 標準化
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 8. 映射四級為 1–4
order4 = ['高中及以下', '大學', '碩士', '博士']
mapping4 = {deg: i+1 for i, deg in enumerate(order4)}
df['degree_ord4'] = df['Degree4'].map(mapping4)

# ——— 分析與繪圖 ———

# A. 計算並保留所有四級，如果某級沒有樣本則為 0%
rate4 = (
    df.groupby('Degree4')['Depression']
    .mean()
    .mul(100)
    .round(1)
    .reindex(order4)
    .fillna(0)
)

plt.figure(figsize=(6, 5))
sns.barplot(x=rate4.index, y=rate4.values, palette='viridis')
plt.xlabel('學歷等級')
plt.ylabel('憂鬱症比例 (%)')
plt.title('學歷等級 vs 憂鬱率')
plt.ylim(0, rate4.max() + 5)
for i, v in enumerate(rate4.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()

# B. 相關係數
corr4 = df['degree_ord4'].corr(df['Depression'])
print(f"Degree4 序數 vs Depression 相關係數：{corr4:.3f}")

# C. 多特徵 Logistic Regression 預測（提升精確度）
features = ['degree_ord4'] + numeric_cols  # 加入所有數值特徵
X = df[features]
y = df['Depression']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 使用 class_weight='balanced' 且增加 max_iter
lr = LogisticRegression(max_iter=2000, class_weight='balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"多特徵 Logistic Regression 準確率：{acc:.3f}，AUC：{auc:.3f}")
