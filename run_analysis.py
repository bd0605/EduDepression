# 版本：v1.0.0
# 學生憂鬱症風險分析 - 簡化版
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
import pymysql
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def load_data():
    """載入資料"""
    df = pd.read_csv('data/student_depression_dataset.csv')
    print(f"✅ 資料載入完成：____ 筆資料")
    return df

def save_to_mysql(df):
    """儲存資料到MySQL"""
    try:
        engine = create_engine('mysql+pymysql://root@localhost/student_depression')
        df.to_sql('depression_data', engine, if_exists='replace', index=False)
        print("✅ 資料已儲存到MySQL")
    except Exception as e:
        print(f"❌ MySQL連接失敗: {e}")

def basic_analysis(df):
    """基礎統計分析"""
    # 學歷與憂鬱症分析
    degree_stats = df.groupby('Degree')['Depression'].agg(['count', 'sum', 'mean']).round(3)
    degree_stats.columns = ['總人數', '憂鬱人數', '憂鬱比例']
    degree_stats['憂鬱百分比'] = (degree_stats['憂鬱比例'] * 100).round(1)
    
    print("\n## 基礎統計分析")
    print("資料筆數：____")
    print("欄位數量：____")
    print("憂鬱症整體比例：____％")
    
    print("\n學歷憂鬱症比例：")
    for degree, row in degree_stats.iterrows():
        print(f"  {degree}: ____％")
    
    return degree_stats

def kmeans_analysis(df):
    """K-means聚類分析"""
    features = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction']
    X = df[features].copy()
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 結果統計
    cluster_stats = df.groupby('Cluster')['Depression'].agg(['count', 'mean'])
    cluster_stats.columns = ['樣本數', '憂鬱比例']
    
    # 視覺化
    plt.figure(figsize=(8, 6))
    plt.bar(cluster_stats.index, cluster_stats['憂鬱比例'], color=['#3498db', '#e74c3c', '#f39c12'])
    plt.xlabel('群組')
    plt.ylabel('憂鬱比例')
    plt.title('K-means 聚類結果', fontsize=16)
    plt.ylim(0, 1)
    plt.savefig('visuals/kmeans_result.png')
    plt.close()
    
    print("\n## K-means聚類分析")
    print("群組數：3")
    for i, row in cluster_stats.iterrows():
        print(f"群組{i}: 樣本數=____, 憂鬱比例=____％")
    
    return cluster_stats

def logistic_regression(df):
    """邏輯回歸分析"""
    features = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction']
    X = df[features]
    y = df['Depression']
    
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 訓練模型
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # 預測
    y_pred = lr.predict(X_test_scaled)
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    
    # 混淆矩陣
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩陣')
    plt.ylabel('實際')
    plt.xlabel('預測')
    plt.savefig('visuals/confusion_matrix.png')
    plt.close()
    
    # ROC曲線
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC曲線 (AUC = ____)')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('假陽性率')
    plt.ylabel('真陽性率')
    plt.title('ROC曲線')
    plt.legend()
    plt.savefig('visuals/roc_curve.png')
    plt.close()
    
    print("\n## 邏輯回歸分析")
    print(f"準確率：____％")
    print(f"AUC值：____")
    print("\n特徵重要性：")
    for feature, coef in zip(features, lr.coef_[0]):
        print(f"  {feature}: 係數=____")
    
    return accuracy, roc_auc

def create_visualizations(df, degree_stats):
    """建立基礎視覺化"""
    # 學歷憂鬱率長條圖
    plt.figure(figsize=(10, 6))
    degrees = degree_stats.index
    depression_rates = degree_stats['憂鬱百分比']
    bars = plt.bar(degrees, depression_rates, color=sns.color_palette("RdYlBu_r", len(degrees)))
    plt.xlabel('學歷', fontsize=12)
    plt.ylabel('憂鬱症比例 (%)', fontsize=12)
    plt.title('不同學歷憂鬱症比例', fontsize=16)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # 加上數值標籤（留空待填寫）
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'__%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('visuals/degree_depression_rate.png')
    plt.close()
    
    print("✅ 視覺化圖表已生成")

def generate_report():
    """生成報告模板"""
    report_template = """
# 學生憂鬱症風險分析報告
## 版本：v1.0.0
## 日期：2025-05-11

## 1. 資料來源與內容
- **資料來源**：Student Depression Dataset
- **資料筆數**：____ 筆
- **資料欄位**：____ 個

### 主要欄位說明：
1. Gender（性別）
2. Age（年齡）
3. Degree（學歷）
4. Academic Pressure（學業壓力）
5. Work Pressure（工作壓力）
6. CGPA（學業成績）
7. Study Satisfaction（學習滿意度）
8. Depression（是否有憂鬱症）

## 2. 資料清洗過程
- 檢查缺失值：____個缺失值
- 處理異常值：____個異常值
- 資料標準化：使用StandardScaler進行特徵標準化

## 3. 分析結果

### 3.1 基礎統計分析
- 整體憂鬱症比例：____％
- 最高風險學歷：____（憂鬱率：____％）
- 最低風險學歷：____（憂鬱率：____％）

### 3.2 K-means聚類分析
- 群組數：3
- 群組0：樣本數=____，憂鬱比例=____％
- 群組1：樣本數=____，憂鬱比例=____％
- 群組2：樣本數=____，憂鬱比例=____％

### 3.3 邏輯回歸分析
- 模型準確率：____％
- AUC值：____
- 重要特徵：
  - Academic Pressure：係數=____
  - Work Pressure：係數=____
  - CGPA：係數=____
  - Study Satisfaction：係數=____

## 4. 結論與建議
（待填寫）

## 5. 資料視覺化（Grafana Dashboard）
- Depression Rate by Degree
- Total Students by Degree
- Overall Depression Rate

---
*報告生成時間：2025-05-11*
"""
    
    with open('report/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_template)
    
    print("✅ 報告模板已生成")

def main():
    print("=== 學生憂鬱症風險分析 (簡化版) ===")
    print("版本：v1.0.0")
    
    # 載入資料
    df = load_data()
    
    # MySQL儲存
    save_to_mysql(df)
    
    # 基礎分析
    degree_stats = basic_analysis(df)
    
    # 視覺化
    create_visualizations(df, degree_stats)
    
    # K-means聚類
    kmeans_analysis(df)
    
    # 邏輯回歸
    logistic_regression(df)
    
    # 生成報告
    generate_report()
    
    print("\n✅ 分析完成！請查看 report/ 目錄中的報告模板並填寫數值")

if __name__ == "__main__":
    main()
