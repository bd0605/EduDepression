import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, 
    classification_report, roc_curve
)
import pymysql
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# 設定圖表風格
plt.style.use('seaborn-v0_8')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']

class StudentDepressionAnalysis:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        print(f"資料載入完成：{self.df.shape[0]} 筆資料")
        self.prepare_data()
    
    def prepare_data(self):
        """資料預處理與特徵工程"""
        # 轉換類別變數
        self.df['Gender_M'] = (self.df['Gender'] == 'Male').astype(int)
        self.df['Suicidal_Thoughts'] = (self.df['Have you ever had suicidal thoughts ?'] == 'Yes').astype(int)
        self.df['Family_History'] = (self.df['Family History of Mental Illness'] == 'Yes').astype(int)
        
        # 處理Financial Stress
        stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
        self.df['Financial_Stress_Num'] = self.df['Financial Stress'].map(stress_map).fillna(2)
        
        # One-hot encoding for Degree
        self.df = pd.get_dummies(self.df, columns=['Degree'], prefix='Degree')
        
        # 選擇特徵
        self.feature_cols = [
            'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
            'Job Satisfaction', 'Work/Study Hours', 'Age', 'Gender_M',
            'Suicidal_Thoughts', 'Family_History', 'Financial_Stress_Num'
        ] + [col for col in self.df.columns if col.startswith('Degree_')]
        
        self.X = self.df[self.feature_cols]
        self.y = self.df['Depression']
    
    def save_to_mysql(self):
        """儲存資料到MySQL"""
        try:
            engine = create_engine('mysql+pymysql://root@localhost/student_depression')
            self.df.to_sql('depression_analysis', engine, if_exists='replace', index=False)
            print("✅ 資料已儲存到MySQL")
        except Exception as e:
            print(f"❌ MySQL連接失敗: {e}")
    
    def perform_kmeans(self, n_clusters=5):
        """K-means聚類分析"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # 視覺化聚類結果
        plt.figure(figsize=(12, 8))
        cluster_stats = self.df.groupby('Cluster')['Depression'].agg(['count', 'mean'])
        cluster_stats.columns = ['Count', 'Depression_Rate']
        
        ax = sns.barplot(x=cluster_stats.index, y=cluster_stats['Depression_Rate'])
        plt.title('Depression Rate by K-means Clusters', fontsize=16, fontweight='bold')
        plt.xlabel('Cluster', fontsize=14)
        plt.ylabel('Depression Rate', fontsize=14)
        
        # 在每個bar上顯示數量
        for i, v in enumerate(cluster_stats['Count']):
            ax.text(i, cluster_stats['Depression_Rate'].iloc[i] + 0.01, 
                    f'n={v}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visuals/kmeans_clusters.png', bbox_inches='tight')
        plt.close()
        
        return cluster_stats
    
    def linear_regression_analysis(self):
        """線性迴歸分析"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 線性迴歸
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        
        # 特徵重要性（迴歸係數）
        coef_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Coefficient': lr.coef_
        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        
        # 視覺化
        plt.figure(figsize=(10, 6))
        sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
        plt.title('Linear Regression Coefficients', fontsize=16, fontweight='bold')
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.tight_layout()
        plt.savefig('visuals/regression_coefficients.png', bbox_inches='tight')
        plt.close()
        
        return coef_df
    
    def logistic_regression_tuned(self):
        """調優的邏輯迴歸（目標：ACC > 0.9）"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 網格搜尋
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # 評估指標
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n邏輯迴歸調優結果：")
        print(f"最佳參數：{grid_search.best_params_}")
        print(f"準確率：{accuracy:.3f}")
        print(f"AUC：{auc:.3f}")
        
        # 混淆矩陣
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Depression', 'Depression'],
                    yticklabels=['No Depression', 'Depression'])
        plt.title('Confusion Matrix - Logistic Regression', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        plt.tight_layout()
        plt.savefig('visuals/confusion_matrix.png', bbox_inches='tight')
        plt.close()
        
        # ROC曲線
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve - Logistic Regression', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('visuals/roc_curve.png', bbox_inches='tight')
        plt.close()
        
        return best_model, accuracy, auc
    
    def generate_grafana_dashboard(self):
        """生成Grafana Dashboard配置"""
        dashboard_config = {
            "dashboard": {
                "title": "Student Depression Analysis",
                "panels": [
                    {
                        "title": "Depression Rate by Degree",
                        "type": "barchart",
                        "gridPos": {"h": 9, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "title": "Risk Factors Heatmap",
                        "type": "heatmap",
                        "gridPos": {"h": 9, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "title": "K-means Clusters",
                        "type": "piechart",
                        "gridPos": {"h": 9, "w": 12, "x": 0, "y": 9}
                    },
                    {
                        "title": "Model Performance",
                        "type": "stat",
                        "gridPos": {"h": 9, "w": 12, "x": 12, "y": 9}
                    }
                ]
            }
        }
        
        import json
        with open('grafana_dashboard.json', 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        return dashboard_config

def main():
    # 執行分析
    analysis = StudentDepressionAnalysis('data/student_depression_dataset.csv')
    
    # 1. 儲存到MySQL
    analysis.save_to_mysql()
    
    # 2. K-means聚類
    cluster_stats = analysis.perform_kmeans()
    print("\nK-means聚類結果：")
    print(cluster_stats)
    
    # 3. 線性迴歸
    coef_df = analysis.linear_regression_analysis()
    print("\n迴歸係數（前10）：")
    print(coef_df)
    
    # 4. 邏輯迴歸調優
    model, acc, auc = analysis.logistic_regression_tuned()
    
    # 5. 生成Grafana配置
    dashboard = analysis.generate_grafana_dashboard()
    print("\n✅ Grafana dashboard配置已生成")
    
    print("\n所有分析完成！圖表已儲存至 visuals/ 目錄")

if __name__ == "__main__":
    main()
