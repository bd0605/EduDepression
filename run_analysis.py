# —— 完整整合版：claude 學生憂鬱症分析 ——

# 1. 系統與字型設定（保留版本B的設定方式）
# !apt-get update -qq
# !apt-get install -y fonts-noto-cjk -qq

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report

# 完善的中文字型設定（版本B的優化）
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font_path)
zh_font = FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

# 2. 資料讀取與基本處理
# 2.1 保留原始資料副本（版本B的方法）
raw_df = pd.read_csv('/content/student_depression_dataset.csv')
if raw_df['Depression'].dtype == object:
    raw_df['Depression'] = raw_df['Depression'].map(
        {'No': 0, 'Yes': 1}).astype(int)

# 2.2 Degree處理與四級分類（兩個版本的共同處理）
deg_mode = raw_df['Degree'].mode().iloc[0]
raw_df['Degree'] = raw_df['Degree'].fillna(deg_mode).astype(str).str.strip()
raw_df.loc[raw_df['Degree'] == '其他', 'Degree'] = deg_mode


def simplify_degree(x):
    x = x.lower()
    if 'phd' in x or '博士' in x:
        return '博士'
    if 'master' in x or 'msc' in x or '碩士' in x:
        return '碩士'
    if any(k in x for k in ['bachelor', 'ba', 'b.sc', 'bsc', 'bcom', 'be', 'mba']):
        return '大學'
    return '高中及以下'


raw_df['Degree4'] = raw_df['Degree'].apply(simplify_degree)
order4 = ['高中及以下', '大學', '碩士', '博士']

# 3. 正式資料處理（整合兩個版本）
df = raw_df.copy().reset_index(drop=True)

# 3.1 更嚴謹的數據清理（版本B的方法）
df = df.drop_duplicates().dropna(
    subset=['Degree', 'Depression']).reset_index(drop=True)
df['Degree'] = df['Degree'].fillna(deg_mode).str.strip()
df.loc[df['Degree'] == '其他', 'Degree'] = deg_mode
df['Degree4'] = df['Degree'].apply(simplify_degree)

# 3.2 數值特徵處理（兩個版本的共同處理）
num_cols = ['Age', 'Academic Pressure',
            'Work Pressure', 'CGPA', 'Study Satisfaction']
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
z = stats.zscore(df[num_cols])
df = df[(np.abs(z) < 3).all(axis=1)].reset_index(drop=True)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 3.3 學歷序數與增強特徵（版本B的優點）
df['degree_ord4'] = df['Degree4'].map({d: i+1 for i, d in enumerate(order4)})
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# 4. 數據分析
# 4.1 學歷等級vs憂鬱率分析（版本A的分析）
rate4 = (
    df.groupby('Degree4')['Depression']
    .mean().mul(100).round(1)
    .reindex(order4).fillna(0)
)

plt.figure(figsize=(7, 5))
ax = sns.barplot(x=list(range(len(order4))), y=rate4.values, palette='viridis')
plt.xticks(ticks=list(range(len(order4))), labels=order4,
           fontproperties=zh_font, fontsize=11)
plt.xlabel('學歷等級', fontproperties=zh_font, fontsize=12)
plt.ylabel('憂鬱症比例 (%)', fontproperties=zh_font, fontsize=12)
plt.title('學歷等級 vs 憂鬱率（清洗後數據）', fontproperties=zh_font, fontsize=14)
plt.ylim(0, rate4.max()+5)

for i, v in enumerate(rate4.values):
    plt.text(i, v+1, f'{v:.1f}%', ha='center',
             fontproperties=zh_font, fontsize=11)

plt.tight_layout()
plt.show()

# 4.2 PCA分析（版本B的優勢功能）
features_all = ['degree_ord4'] + \
    [c for c in df.columns if c.startswith('Gender_')] + num_cols
pca = PCA(n_components=min(5, len(features_all)))
pca.fit(df[features_all])
load = pd.Series(pca.components_[0], index=features_all).abs(
).sort_values(ascending=False)
print("\nPC1 解釋變異比例：", np.round(pca.explained_variance_ratio_[0], 3))
print("PC1 載荷量排序：\n", load.round(3))
print("→ 最重要特徵：", load.index[0])

# 4.3 相關係數分析（兩個版本共有）
corr4 = df['degree_ord4'].corr(df['Depression'])
print(f"\ndegree_ord4 vs Depression 相關係數：{corr4:.3f}")

# 5. 模型建立與評估
# 5.1 準備特徵與標籤
features = ['degree_ord4'] + \
    [c for c in df.columns if c.startswith('Gender_')] + num_cols
X, y = df[features], df['Depression']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5.2 Logistic Regression模型（兩個版本共有）
lr = LogisticRegression(class_weight='balanced', max_iter=2000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

# 5.3 混淆矩陣分析（版本B的優勢）
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['預測 0（無憂鬱）', '預測 1（有憂鬱）'],
            yticklabels=['實際 0（無憂鬱）', '實際 1（有憂鬱）'],
            annot_kws={'fontproperties': zh_font, 'fontsize': 11})
plt.title('Logistic Regression 混淆矩陣', fontproperties=zh_font, fontsize=14)
plt.ylabel('實際標籤', fontproperties=zh_font, fontsize=12)
plt.xlabel('預測標籤', fontproperties=zh_font, fontsize=12)
plt.xticks(fontproperties=zh_font, rotation=0)
plt.yticks(fontproperties=zh_font, rotation=0)
plt.tight_layout()
plt.show()

# 5.4 模型性能詳細評估（新增增強部分）
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"\n多特徵 Logistic Regression 準確率：{acc:.3f}，AUC：{auc:.3f}")
print("\n分類報告：")
print(classification_report(y_test, y_pred))

# 5.5 ROC曲線（新增增強部分）
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假陽性率 (FPR)', fontproperties=zh_font, fontsize=12)
plt.ylabel('真陽性率 (TPR)', fontproperties=zh_font, fontsize=12)
plt.title('ROC曲線', fontproperties=zh_font, fontsize=14)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5.6 特徵重要性分析（新增增強部分）
importance = pd.Series(
    np.abs(lr.coef_[0]), index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
importance[:10].plot(kind='bar')
plt.title('Logistic Regression 特徵重要性（前10項）',
          fontproperties=zh_font, fontsize=14)
plt.xlabel('特徵', fontproperties=zh_font, fontsize=12)
plt.ylabel('係數絕對值', fontproperties=zh_font, fontsize=12)
plt.xticks(rotation=45, ha='right', fontproperties=zh_font)
plt.tight_layout()
plt.show()
