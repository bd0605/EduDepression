# —— 完整整合版：學業壓力與憂鬱風險相關性分析 ——
# 1. 系統與字型設定
from scipy.stats import chi2_contingency
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
import pandas as pd
import os
import platform
import warnings

warnings.filterwarnings(
    "ignore",
    message="Glyph .* missing from font\\(s\\) DejaVu Sans\\.",
    category=UserWarning,
    module="seaborn.utils"
)


# 📦 若在 Colab 上，自動安裝中文字型
try:
    if "google.colab" in str(get_ipython()):
        print("🔧 Colab 偵測中，嘗試安裝中文字型...")
        os.system("apt-get install -y fonts-noto-cjk > /dev/null")
except:
    pass

# ✅ 跨平台自動選擇中文字型


def load_chinese_font():
    system = platform.system()
    paths = []

    if system == "Windows":
        paths = [
            "C:/Windows/Fonts/msjh.ttc",
            "C:/Windows/Fonts/mingliu.ttc"
        ]
    elif system == "Darwin":  # macOS
        paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc"
        ]
    else:  # Linux / Colab
        paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ]

    for path in paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font = FontProperties(fname=path)
            plt.rcParams["font.family"] = font.get_name()
            print(f"✅ 已載入中文字型：{font.get_name()}")
            return font

    print("⚠️ 找不到中文字型，使用預設英文字型")
    return FontProperties()


# ✅ 設定全域 zh_font
zh_font = load_chinese_font()
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


# 2. 資料讀取與基本處理
# 2.1 保留原始資料副本
raw_df = pd.read_csv('/content/student_depression_dataset.csv')
print(f"原始資料集大小: {raw_df.shape}")
print(f"資料集列名: {raw_df.columns.tolist()}")

if raw_df['Depression'].dtype == object:
    raw_df['Depression'] = raw_df['Depression'].map(
        {'No': 0, 'Yes': 1}).astype(int)

# 顯示基本資料統計
print("\n基本資料統計 (數值型變數):")
print(raw_df[['Age', 'Academic Pressure', 'Work Pressure',
      'CGPA', 'Study Satisfaction']].describe())

# 2.2 處理學業壓力變數 - 確保正確讀取
if 'Academic Pressure' not in raw_df.columns:
    # 檢查是否有類似名稱的欄位
    possible_columns = [col for col in raw_df.columns if 'pressure' in col.lower(
    ) or 'academic' in col.lower()]
    if possible_columns:
        print(f"找到可能的學業壓力相關欄位: {possible_columns}")
        # 假設第一個是學業壓力欄位
        raw_df['Academic Pressure'] = raw_df[possible_columns[0]]
    else:
        print("未找到學業壓力相關欄位，請檢查資料集")

# 2.3 Degree處理與四級分類
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

# 3. 正式資料處理
df = raw_df.copy().reset_index(drop=True)

# 3.1 學業壓力分類 - 新增
# 確保學業壓力是數值
if 'Academic Pressure' in df.columns:
    if df['Academic Pressure'].dtype == object:
        # 如果是類別資料，轉換為數值
        ap_values = {k: i+1 for i,
                     k in enumerate(sorted(df['Academic Pressure'].unique()))}
        df['Academic Pressure_Value'] = df['Academic Pressure'].map(ap_values)
        # 保留原始類別以便分析
        df['Academic Pressure_Category'] = df['Academic Pressure']
    else:
        # 如果已經是數值，進行分組
        df['Academic Pressure_Value'] = df['Academic Pressure']
        # 使用 qcut 可能導致空值或不均勻分布的問題，改用 cut
        # 檢查數據的分佈
        ap_min = df['Academic Pressure_Value'].min()
        ap_max = df['Academic Pressure_Value'].max()

        # 使用 cut 進行分組，確保每個組有數據
        bins = [ap_min, ap_min + (ap_max-ap_min)/3,
                ap_min + 2*(ap_max-ap_min)/3, ap_max]
        df['Academic Pressure_Category'] = pd.cut(
            df['Academic Pressure_Value'],
            bins=bins,
            labels=['低壓力', '中壓力', '高壓力'],
            include_lowest=True
        )
else:
    print("警告: 資料集中找不到 Academic Pressure 欄位!")
    # 創建一個替代欄位
    df['Academic Pressure_Value'] = df['Work Pressure']  # 假設工作壓力可作為替代
    df['Academic Pressure_Category'] = pd.cut(
        df['Academic Pressure_Value'],
        bins=3,
        labels=['低壓力', '中壓力', '高壓力']
    )

# 3.2 更嚴謹的數據清理
df = df.drop_duplicates().dropna(
    subset=['Academic Pressure_Value', 'Depression']).reset_index(drop=True)

# 3.3 數值特徵處理
num_cols = ['Age', 'Academic Pressure_Value',
            'Work Pressure', 'CGPA', 'Study Satisfaction']
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
z = stats.zscore(df[num_cols])
df = df[(np.abs(z) < 3).all(axis=1)].reset_index(drop=True)
# scaler = StandardScaler()
# df_scaled = pd.DataFrame(
#     scaler.fit_transform(df[num_cols]),
#     columns=num_cols,
#     index=df.index
# )

# 3.4 學歷序數與增強特徵
df['degree_ord4'] = df['Degree4'].map({d: i+1 for i, d in enumerate(order4)})
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# 4. 學業壓力與憂鬱風險相關性分析

# 4.1 學業壓力與憂鬱症的關聯分析 - 新增
print("\n====== 學業壓力與憂鬱風險相關性分析 ======")
ap_corr = df['Academic Pressure_Value'].corr(df['Depression'])
print(f"學業壓力與憂鬱症的相關係數: {ap_corr:.3f}")

# 4.2 不同壓力水平的憂鬱風險分析 - 新增
ap_group = df.groupby('Academic Pressure_Category', observed=False)[
    'Depression'].agg(['mean', 'count'])

ap_group.columns = ['憂鬱比例', '樣本數']
print("\n不同學業壓力水平的憂鬱風險:")
print(ap_group)

# 診斷和修復圖表顯示問題
print("\n====== 診斷圖表顯示問題 ======")

# 檢查原始資料
print(f"原始資料大小: {df.shape}")
print(
    f"Academic Pressure_Value 範圍: {df['Academic Pressure_Value'].min()} 到 {df['Academic Pressure_Value'].max()}")
print(f"憂鬱變數 (Depression) 值計數: \n{df['Depression'].value_counts()}")

# 檢查分組資料
print("\n學業壓力分組情況:")
ap_category_counts = df['Academic Pressure_Category'].value_counts()
print(ap_category_counts)

# 檢查低、中、高壓力組的憂鬱比例
print("\n各壓力組的憂鬱比例:")
for category in df['Academic Pressure_Category'].unique():
    subset = df[df['Academic Pressure_Category'] == category]
    depression_rate = subset['Depression'].mean()
    count = len(subset)
    print(f"{category}: 憂鬱比例 = {depression_rate:.4f}, 樣本數 = {count}")

# 使用最基本的方式確保視覺化能夠工作
plt.figure(figsize=(16, 6))

# === 第一個圖：最簡單的條形圖 ===
plt.subplot(1, 2, 1)

# 直接創建三個分類和對應的憂鬱比例
categories = ['低壓力', '中壓力', '高壓力']
depression_rates = []

# 手動計算各類別的憂鬱比例
for category in categories:
    if category in df['Academic Pressure_Category'].values:
        subset = df[df['Academic Pressure_Category'] == category]
        rate = subset['Depression'].mean()
    else:
        rate = 0  # 如果類別不存在，設為0
    depression_rates.append(rate)

# 打印檢查資料
print("\n用於繪圖的資料:")
for cat, rate in zip(categories, depression_rates):
    print(f"{cat}: {rate:.4f}")

# 使用最基本的plt.bar繪製條形圖
bars = plt.bar(categories, depression_rates, color=['blue', 'green', 'red'])

# 添加數據標籤
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.02,
        f'{height:.2f}',
        ha='center',
        fontsize=12
    )

plt.xticks(fontproperties=zh_font, fontsize=14)
plt.yticks(fontproperties=zh_font, fontsize=14)
plt.xlabel('學業壓力水平', fontproperties=zh_font)
plt.ylabel('憂鬱比例', fontproperties=zh_font)
plt.title('不同學業壓力水平的憂鬱比例', fontproperties=zh_font)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# === 第二個圖：學業壓力連續值的分組柱狀圖 ===
plt.subplot(1, 2, 2)

# 創建簡單的分組 - 只分成5組以確保每組有足夠資料
n_bins = 5
min_val = df['Academic Pressure_Value'].min()
max_val = df['Academic Pressure_Value'].max()
bin_edges = np.linspace(min_val, max_val, n_bins + 1)

# 創建分組標籤
bin_labels = [
    f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(n_bins)]

# 手動分組並計算每組的憂鬱比例
depression_by_bin = []
counts_by_bin = []

for i in range(n_bins):
    if i < n_bins - 1:
        mask = (df['Academic Pressure_Value'] >= bin_edges[i]) & (
            df['Academic Pressure_Value'] < bin_edges[i+1])
    else:  # 最後一組包含右邊界
        mask = (df['Academic Pressure_Value'] >= bin_edges[i]) & (
            df['Academic Pressure_Value'] <= bin_edges[i+1])

    group_data = df[mask]
    count = len(group_data)
    counts_by_bin.append(count)

    if count > 0:
        depression_rate = group_data['Depression'].mean()
    else:
        depression_rate = 0

    depression_by_bin.append(depression_rate)

# 打印用於繪圖的資料
print("\n學業壓力連續值分組:")
for label, rate, count in zip(bin_labels, depression_by_bin, counts_by_bin):
    print(f"{label}: 憂鬱比例 = {rate:.4f}, 樣本數 = {count}")

# 確保所有數值有效，避免空白圖表
if all(count > 0 for count in counts_by_bin) and any(rate > 0 for rate in depression_by_bin):
    # 使用最基本的plt.bar繪製條形圖
    bars = plt.bar(bin_labels, depression_by_bin, color=[
                   'skyblue', 'lightgreen', 'lightsalmon', 'lightpink', 'lightgoldenrodyellow'])

    # 添加數據標籤
    for bar, count in zip(bars, counts_by_bin):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f'{height:.2f}\nn={count}',
            ha='center',
            fontsize=10
        )

    plt.title('學業壓力指數與憂鬱風險關係', fontproperties=zh_font, fontsize=16)
    plt.xlabel('學業壓力指數區間', fontproperties=zh_font, fontsize=14)
    plt.ylabel('憂鬱比例', fontproperties=zh_font, fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=30, fontproperties=zh_font, ha='right')
else:
    plt.text(0.5, 0.5, '無足夠資料生成圖表',
             horizontalalignment='center',
             verticalalignment='center',
             fontproperties=zh_font, fontsize=16)
    print("警告: 無法生成圖表，因為資料分組後樣本數不足或憂鬱比例全為0")

plt.tight_layout()

# ❶ 定義 contingency_table
contingency_table = pd.crosstab(
    df['Academic Pressure_Category'],
    df['Depression']
)

print("\n交叉列聯表：")
print(contingency_table)

# 使用卡方檢定
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print("\n不同壓力等級的憂鬱風險差異檢定 (卡方檢定):")
print(f"卡方值: {chi2:.3f}, 自由度: {dof}, p-value: {p_value:.4f}")
print(f"結論: {'壓力等級之間憂鬱風險有顯著差異' if p_value < 0.05 else '壓力等級之間憂鬱風險沒有顯著差異'}")

# 4.6 PCA分析（考慮學業壓力）
features_all = ['degree_ord4', 'Academic Pressure_Value'] + \
    [c for c in df.columns if c.startswith('Gender_')] + num_cols
pca = PCA(n_components=min(5, len(features_all)))
pca.fit(df[features_all])
load = pd.Series(pca.components_[0], index=features_all).abs(
).sort_values(ascending=False)
print("\nPCA分析（主成分分析）:")
print("PC1 解釋變異比例：", np.round(pca.explained_variance_ratio_[0], 3))
print("PC1 載荷量排序（前5項）：\n", load.head(5).round(3))
print("→ 學業壓力在PC1中的重要性排名：", list(load.index).index('Academic Pressure_Value') + 1)

# 5. 學業壓力特定分析 - 新增

# 5.1 學業壓力與其他變數的交互作用
print("\n====== 學業壓力與其他變數的交互作用 ======")

# 計算學業壓力與其他數值變數的相關性
corr_with_ap = df[num_cols].corr(
)['Academic Pressure_Value'].sort_values(ascending=False)
print("學業壓力與其他變數的相關性:")
print(corr_with_ap)

# 6. 模型建立與評估
scaler = StandardScaler()  # ← ✅ 加上這行就不會報 NameError

# 6.1 準備特徵與標籤，並標準化
features = ['Academic Pressure_Value', 'degree_ord4'] + [c for c in df.columns if c.startswith(
    'Gender_')] + [col for col in num_cols if col != 'Academic Pressure_Value']
X = scaler.fit_transform(df[features])
y = df['Depression']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6.2 Logistic Regression模型
print("\n====== 預測模型建立與評估 ======")
print("\nLogistic Regression模型:")
lr = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',    # 用 liblinear 演算法更穩定
    max_iter=5000,         # 多跑幾次迭代
    random_state=42
)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]  # type: ignore[index]

# 6.3 混淆矩陣分析
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['預測 0（無憂鬱）', '預測 1（有憂鬱）'],  # type: ignore[arg-type]
    yticklabels=['實際 0（無憂鬱）', '實際 1（有憂鬱）'],  # type: ignore[arg-type]
    annot_kws={'fontproperties': zh_font, 'fontsize': 11}
)
plt.title('Logistic Regression 混淆矩陣', fontproperties=zh_font, fontsize=14)
plt.ylabel('實際標籤', fontproperties=zh_font, fontsize=12)
plt.xlabel('預測標籤', fontproperties=zh_font, fontsize=12)
plt.xticks(fontproperties=zh_font, rotation=0)
plt.yticks(fontproperties=zh_font, rotation=0)
plt.tight_layout()
plt.show()

# 6.4 模型性能詳細評估
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"多特徵 Logistic Regression 準確率：{acc:.3f}，AUC：{auc:.3f}")
print("\n分類報告：")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0        # 預測不到某類別時回傳 0，不出 warning
))

# 6.5 特徵重要性分析
importance = pd.Series(
    np.abs(lr.coef_[0]), index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importance.plot(kind='bar')
plt.title('Logistic Regression 特徵重要性', fontproperties=zh_font, fontsize=14)
plt.xlabel('特徵', fontproperties=zh_font, fontsize=12)
plt.ylabel('係數絕對值', fontproperties=zh_font, fontsize=12)
plt.xticks(rotation=45, ha='right', fontproperties=zh_font)
plt.tight_layout()
plt.show()

# 6.6 隨機森林模型與特徵重要性比較 - 新增
print("\n隨機森林模型:")
rf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)
print(f"隨機森林模型準確率: {rf_acc:.3f}, AUC: {rf_auc:.3f}")

# 計算隨機森林特徵重要性
result = permutation_importance(      # type: ignore[call-arg]
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
rf_importance = pd.Series(
    result.importances_mean,          # type: ignore[attr-defined]
    index=features
).sort_values(ascending=False)

# 6.7 兩種模型的學業壓力重要性比較
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
importance.head(5).plot(kind='bar')
plt.title('LR模型特徵重要性 (Top 5)', fontproperties=zh_font, fontsize=14)
plt.xlabel('特徵', fontproperties=zh_font, fontsize=12)
plt.ylabel('係數絕對值', fontproperties=zh_font, fontsize=12)
plt.xticks(rotation=45, ha='right', fontproperties=zh_font)

plt.subplot(1, 2, 2)
rf_importance.head(5).plot(kind='bar')
plt.title('RF模型特徵重要性 (Top 5)', fontproperties=zh_font, fontsize=14)
plt.xlabel('特徵', fontproperties=zh_font, fontsize=12)
plt.ylabel('重要性分數', fontproperties=zh_font, fontsize=12)
plt.xticks(rotation=45, ha='right', fontproperties=zh_font)

plt.tight_layout()
plt.show()

# 6.8 ROC曲線比較 - 新增
plt.figure(figsize=(8, 6))
# Logistic Regression ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'LR (AUC = {auc:.3f})')

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
plt.plot(rf_fpr, rf_tpr, label=f'RF (AUC = {rf_auc:.3f})')

# 對角線
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('偽陽性率 (FPR)', fontproperties=zh_font, fontsize=12)
plt.ylabel('真陽性率 (TPR)', fontproperties=zh_font, fontsize=12)
plt.title('預測憂鬱風險的ROC曲線比較', fontproperties=zh_font, fontsize=14)
plt.legend(loc="lower right", prop=zh_font)
plt.grid(True)
plt.show()

# 7. 研究結論與建議 - 新增
print("\n====== 研究結論與建議 ======")
# 學業壓力的重要性評估
ap_lr_rank = list(importance.index).index('Academic Pressure_Value') + \
    1 if 'Academic Pressure_Value' in importance.index else "未在特徵中"
ap_rf_rank = list(rf_importance.index).index('Academic Pressure_Value') + \
    1 if 'Academic Pressure_Value' in rf_importance.index else "未在特徵中"

print(f"\n1. 學業壓力與憂鬱風險的相關係數為 {ap_corr:.3f}")
if ap_corr > 0:
    print("   學業壓力越高，憂鬱風險傾向於增加")
elif ap_corr < 0:
    print("   學業壓力越高，憂鬱風險傾向於減少")
else:
    print("   學業壓力與憂鬱風險沒有明顯的線性關係")

print(f"\n2. 學業壓力在預測憂鬱風險的特徵中:")
print(f"   - 在Logistic Regression模型中排名第{ap_lr_rank}位")
print(f"   - 在隨機森林模型中排名第{ap_rf_rank}位")

print("\n3. 不同學業壓力級別的憂鬱風險:")
for level, row in ap_group.iterrows():
    if isinstance(row, pd.Series):
        print(f"   - {level}: {row['憂鬱比例']:.2%} (樣本數: {row['樣本數']})")

print("\n4. 研究建議:")
print("   - 根據分析結果，應該提供更多的心理健康資源給高學業壓力學生")
print("   - 學校可以發展壓力管理培訓課程，特別針對高風險學生群體")
print("   - 進一步研究可以探索學業壓力與其他因素（如社交支持、睡眠質量）的交互作用")
