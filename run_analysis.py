# -*- coding: utf-8 -*-
"""
學業壓力與憂鬱風險相關性分析主控程式

此程式整合所有分析流程，從資料前處理、模型訓練到視覺化，
並支援資料匯出至 MySQL 以供 Grafana 視覺化使用。

用法：
    python run_analysis.py [--to-mysql]

參數：
    --to-mysql: 是否匯出資料至 MySQL，預設為否
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import chi2_contingency
import time
import threading

# 載入自定義模組
from src.preprocess import preprocess
from src.plot_utils import (
    setup_chinese_font,
    plot_combined_depression_charts,
    plot_confusion_matrix,
    plot_feature_importance_bar,
    plot_roc_curves
)
from src.model_utils import train_and_evaluate
from src.db_utils import (
    test_connection,
    export_to_mysql,
    create_schema
)
from src.font_loader import download_font_if_not_exist

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class RunningIndicator:
    """簡單的運行指示器"""
    
    def __init__(self, message="系統運行中，請稍後"):
        self.message = message
        self.running = False
        self.thread = None
    
    def start(self):
        """開始顯示運行指示器"""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """停止運行指示器"""
        self.running = False
        if self.thread:
            self.thread.join()
        print()  # 換行
    
    def _animate(self):
        """運行動畫"""
        chars = "|/-\\"
        idx = 0
        while self.running:
            print(f"\r{self.message} {chars[idx % len(chars)]}", end="", flush=True)
            idx += 1
            time.sleep(0.3)

def setup_environment():
    """
    設定環境與字型
    
    Returns:
        matplotlib.font_manager.FontProperties: 中文字型屬性物件
    """
    print("🚀 EduDepression 學業壓力與憂鬱風險分析系統")
    print("=" * 60)
    
    # 只在這裡顯示運行指示器
    indicator = RunningIndicator("🔧 系統初始化中，請稍後")
    indicator.start()
    
    try:
        # 下載並設定中文字型
        font_path = download_font_if_not_exist()
        zh_font = setup_chinese_font(font_path)
        
        # 強制設定 matplotlib 和 seaborn 的中文字型
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
        # 設定 seaborn 字型
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.0)
        
        indicator.stop()
        print("✅ 系統初始化完成")
        print("")
        
        return zh_font
        
    except Exception as e:
        indicator.stop()
        print(f"❌ 系統初始化失敗: {e}")
        sys.exit(1)

def parse_args():
    """
    解析命令列參數
    
    Returns:
        argparse.Namespace: 解析後的參數
    """
    parser = argparse.ArgumentParser(description='學業壓力與憂鬱風險相關性分析')
    parser.add_argument('--to-mysql', action='store_true',
                        help='匯出分析結果至 MySQL 資料庫')
    return parser.parse_args()

def run_data_preprocessing(args):
    """
    讀取並前處理資料

    Args:
        args: 命令列參數
        
    Returns:
        pandas.DataFrame: 處理後的資料框
    """
    print("📊 資料預處理")
    print("-" * 40)
    
    try:
        # 讀取資料
        data_path = 'data/student_depression_dataset.csv'
        df = preprocess(data_path)
        
        print(f"✅ 資料讀取成功")
        print(f"   處理後資料集大小: {df.shape[0]:,} 筆資料, {df.shape[1]} 個特徵")
        print("")
        
        return df
        
    except Exception as e:
        print(f"❌ 資料處理失敗: {e}")
        sys.exit(1)

def run_statistical_analysis(df, zh_font):
    """
    執行統計分析

    Args:
        df (pandas.DataFrame): 處理後的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
    """
    print("📈 統計分析")
    print("-" * 40)
    
    # 計算學業壓力與憂鬱風險的相關係數
    ap_corr = df['Academic Pressure_Value'].corr(df['Depression'])
    
    # 分析各壓力組的憂鬱比例
    ap_group = df.groupby('Academic Pressure_Category')['Depression'].agg(['mean', 'count'])
    ap_group.columns = ['憂鬱比例', '樣本數']
    
    # 創建交叉列聯表與卡方檢定
    contingency_table = pd.crosstab(
        df['Academic Pressure_Category'],
        df['Depression']
    )
    
    # 執行卡方檢定
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # 繪製視覺化圖表
    plot_combined_depression_charts(df, zh_font)
    
    print(f"✅ 相關性分析完成")
    print(f"   學業壓力與憂鬱風險相關係數: {ap_corr:.3f}")
    print("")
    print("   各壓力水平憂鬱風險統計:")
    
    # 顯示各壓力組詳細統計
    for category in ['低壓力', '中壓力', '高壓力']:
        if category in df['Academic Pressure_Category'].values:
            subset = df[df['Academic Pressure_Category'] == category]
            depression_rate = subset['Depression'].mean()
            count = len(subset)
            print(f"     {category}: {depression_rate:.2%} (樣本數: {count:,})")
    
    print("")
    print(f"✅ 卡方獨立性檢定完成")
    print(f"   卡方值: {chi2:.3f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   結論: {'壓力等級間憂鬱風險有顯著差異 (p < 0.05)' if p_value < 0.05 else '壓力等級間憂鬱風險無顯著差異'}")
    print("")

def run_model_analysis(df, zh_font):
    """
    執行模型訓練與評估

    Args:
        df (pandas.DataFrame): 處理後的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        
    Returns:
        dict: 模型訓練與評估結果
    """
    print("🤖 機器學習模型")
    print("-" * 40)
    
    # 選擇特徵
    features = [
        'Academic Pressure_Value', 'degree_ord4', 'Age', 
        'CGPA', 'Study Satisfaction'
    ]
    features.extend([col for col in df.columns if col.startswith('Gender_')])
    
    # 訓練與評估模型
    results = train_and_evaluate(df, features)

    # 繪製模型評估圖表
    # 混淆矩陣
    plot_confusion_matrix(
        results['lr_results']['confusion_matrix'], 
        zh_font, 
        'Logistic Regression 混淆矩陣'
    )
    
    # 特徵重要性圖表
    plot_feature_importance_bar(
        results['lr_importance'], 
        zh_font, 
        'Logistic Regression 特徵重要性'
    )
    plot_feature_importance_bar(
        results['rf_importance'], 
        zh_font, 
        'Random Forest 特徵重要性'
    )
    
    # ROC 曲線
    y_test = results['y_test']
    roc_data = [
    (y_test, results['lr_results']['y_proba'], 'Logistic Regression', results['lr_results']['auc']),
    (y_test, results['rf_results']['y_proba'], 'Random Forest',       results['rf_results']['auc'])
    ]
    plot_roc_curves(roc_data, zh_font)
    
    print("✅ 模型訓練完成")
    print(f"   Logistic Regression 準確率: {results['lr_results']['accuracy']:.1%}")
    print(f"   Random Forest 準確率: {results['rf_results']['accuracy']:.1%}")
    print("")
    print(f"   Logistic Regression AUC: {results['lr_results']['auc']:.3f}")
    print(f"   Random Forest AUC: {results['rf_results']['auc']:.3f}")
    print("")
    
    # 輸出研究結論
    print("📋 研究結論")
    print("-" * 40)
    print(f"1. 學業壓力與憂鬱風險相關係數: {df['Academic Pressure_Value'].corr(df['Depression']):.3f}")
    
    print(f"2. 學業壓力在預測特徵中的重要性排名:")
    print(f"   • Logistic Regression: 第 {results['ap_lr_rank']} 位")
    print(f"   • Random Forest: 第 {results['ap_rf_rank']} 位")
    
    print("3. 不同壓力級別的憂鬱風險:")
    for level, row in df.groupby('Academic Pressure_Category')['Depression'].agg(['mean', 'count']).iterrows():
        print(f"   • {level}: {row['mean']:.2%} (樣本數: {row['count']:,})")
    
    print("4. 建議事項:")
    print("   • 針對高壓力學生提供心理健康資源")
    print("   • 發展壓力管理培訓課程")
    print("   • 探索壓力與其他因素的交互作用")
    print("")
    
    return results

def export_data_to_mysql(df):
    """
    將資料匯出至 MySQL 資料庫

    Args:
        df (pandas.DataFrame): 處理後的資料框
        
    Returns:
        bool: 匯出是否成功
    """
    print("🗄️ 資料庫匯出")
    print("-" * 40)
    
    # 測試資料庫連接
    if not test_connection():
        print("❌ 無法連接至 MySQL 資料庫")
        print("   請檢查 XAMPP 是否啟動，MySQL 服務是否正常")
        return False
    
    # 建立資料庫結構
    sql_file_path = os.path.join(os.path.dirname(__file__), "db", "create_table.sql")
    if os.path.exists(sql_file_path):
        create_schema(sql_file_path)
    
    # 匯出資料
    success = export_to_mysql(df, "student_depression")
    
    # 回報結果
    if success:
        print("✅ 資料成功匯出至 MySQL")
        print("   您現在可以使用 Grafana 連接 MySQL 進行視覺化")
    else:
        print("❌ 資料匯出失敗")
    
    print("")
    return success

def main():
    """
    主程式流程
    """
    # 解析命令列參數
    args = parse_args()
    
    try:
        # 設定環境與字型
        zh_font = setup_environment()
        
        # 讀取並前處理資料
        df = run_data_preprocessing(args)
        
        # 執行統計分析
        run_statistical_analysis(df, zh_font)
        
        # 執行模型訓練與評估
        model_results = run_model_analysis(df, zh_font)
        
        # 匯出資料至 MySQL（如果指定）
        if args.to_mysql:
            export_data_to_mysql(df)
        else:
            print("💡 提示")
            print("-" * 40)
            print("若要匯出資料至 MySQL，請使用 --to-mysql 參數")
            print("範例: python run_analysis.py --to-mysql")
            print("")
        
        # 完成分析
        print("🎉 分析完成")
        print("=" * 60)
        print("所有分析已成功完成！圖表已儲存，可查看視覺化結果。")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷了分析程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 分析過程中發生錯誤: {e}")
        sys.exit(1)

# 當直接執行此模組時
if __name__ == "__main__":
    # 防止 plt.show() 阻塞程式執行
    plt.ion()
    
    # 執行主程式
    main()
    
    # 等待使用者按鍵結束程式
    plt.ioff()
    input("\n按下 Enter 鍵結束程式...")
