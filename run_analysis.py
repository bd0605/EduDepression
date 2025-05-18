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
from scipy.stats import chi2_contingency

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
from src.db_utils import export_to_mysql, test_connection, create_schema
from src.font_loader import download_font_if_not_exist

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="Glyph .* missing from font\\(s\\) DejaVu Sans\\.",
    category=UserWarning,
    module="seaborn.utils"
)

def setup_environment():
    """
    設定執行環境，包含字型設定與警告過濾等

    Returns:
        matplotlib.font_manager.FontProperties: 中文字型屬性物件
    """
    font_path = download_font_if_not_exist()
    fm.fontManager.addfont(font_path)

    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus'] = False

    return FontProperties(fname=font_path)

def parse_args():
    """
    解析命令列參數

    Returns:
        argparse.Namespace: 解析後的參數物件
    """
    parser = argparse.ArgumentParser(description='學業壓力與憂鬱風險相關性分析')
    parser.add_argument('--to-mysql', action='store_true', 
                       help='是否匯出資料至 MySQL，預設為否')
    parser.add_argument('--data-path', type=str, 
                       default='data/student_depression_dataset.csv',
                       help='資料集路徑')
    return parser.parse_args()

def run_basic_analysis(df, zh_font):
    """
    執行基本資料分析

    Args:
        df (pandas.DataFrame): 處理後的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
    """
    # 顯示資料集基本統計
    print(f"處理後資料集大小: {df.shape}")
    
    # 計算學業壓力與憂鬱風險的相關係數
    ap_corr = df['Academic Pressure_Value'].corr(df['Depression'])
    print(f"學業壓力與憂鬱風險的相關係數: {ap_corr:.3f}")
    
    # 分析各壓力組的憂鬱比例
    ap_group = df.groupby('Academic Pressure_Category')[
        'Depression'].agg(['mean', 'count'])
    ap_group.columns = ['憂鬱比例', '樣本數']
    print("\n不同學業壓力水平的憂鬱風險:")
    print(ap_group)
    
    # 顯示低、中、高壓力組的憂鬱比例
    for category in ['低壓力', '中壓力', '高壓力']:
        if category in df['Academic Pressure_Category'].values:
            subset = df[df['Academic Pressure_Category'] == category]
            depression_rate = subset['Depression'].mean()
            count = len(subset)
            print(f"{category}: 憂鬱比例 = {depression_rate:.4f}, 樣本數 = {count}")
    
    # 創建交叉列聯表
    contingency_table = pd.crosstab(
        df['Academic Pressure_Category'],
        df['Depression']
    )
    print("\n交叉列聯表：")
    print(contingency_table)
    
    # 執行卡方檢定
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print("\n不同壓力等級的憂鬱風險差異檢定 (卡方檢定):")
    print(f"卡方值: {chi2:.3f}, 自由度: {dof}, p-value: {p_value:.4f}")
    print(f"結論: {'壓力等級之間憂鬱風險有顯著差異' if p_value < 0.05 else '壓力等級之間憂鬱風險沒有顯著差異'}")
    
    # 繪製視覺化圖表
    plot_combined_depression_charts(df, zh_font)

def export_to_db(df):
    """
    將資料匯出至 MySQL 資料庫

    Args:
        df (pandas.DataFrame): 處理後的資料框
        
    Returns:
        bool: 匯出是否成功
    """
    # 測試資料庫連接
    if not test_connection():
        print("無法連接至 MySQL 資料庫，請檢查連接設定")
        return False
    
    # 建立資料庫結構
    sql_file_path = os.path.join(os.path.dirname(__file__), "db", "create_table.sql")
    if os.path.exists(sql_file_path):
        print("建立資料庫結構...")
        create_schema(sql_file_path)
    
    # 匯出資料
    print("匯出資料至 MySQL...")
    success = export_to_mysql(df, "student_depression")
    
    # 回報結果
    if success:
        print("資料成功匯出至 MySQL！")
        print("您現在可以使用 Grafana 連接 MySQL 進行視覺化")
    else:
        print("資料匯出失敗！")
    
    return success

def run_model_analysis(df, zh_font):
    """
    執行模型訓練與評估

    Args:
        df (pandas.DataFrame): 處理後的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        
    Returns:
        dict: 模型訓練與評估結果
    """
    # 選擇特徵
    features = [
        'Academic Pressure_Value', 'degree_ord4', 'Age', 
        'CGPA', 'Study Satisfaction'
    ]
    features.extend([col for col in df.columns if col.startswith('Gender_')])
    
    # 訓練與評估模型
    print("\n====== 預測模型建立與評估 ======")
    results = train_and_evaluate(df, features)
    
    # 繪製混淆矩陣
    plot_confusion_matrix(
        results['lr_results']['confusion_matrix'], 
        zh_font, 
        'Logistic Regression 混淆矩陣'
    )
    
    # 繪製特徵重要性圖表
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
    
    # 繪製 ROC 曲線
    y_test = results['y_test']                # 直接拿 train_and_evaluate 回傳的 y_test
    roc_data = [
    (y_test, results['lr_results']['y_proba'], 'Logistic Regression', results['lr_results']['auc']),
    (y_test, results['rf_results']['y_proba'], 'Random Forest',       results['rf_results']['auc'])
    ]
    plot_roc_curves(roc_data, zh_font)
    
    # 輸出結論
    print("\n====== 研究結論與建議 ======")
    print(f"\n1. 學業壓力與憂鬱風險的相關係數為 {df['Academic Pressure_Value'].corr(df['Depression']):.3f}")
    
    print(f"\n2. 學業壓力在預測憂鬱風險的特徵中:")
    print(f"   - 在Logistic Regression模型中排名第{results['ap_lr_rank']}位")
    print(f"   - 在隨機森林模型中排名第{results['ap_rf_rank']}位")
    
    print("\n3. 不同學業壓力級別的憂鬱風險:")
    for level, row in df.groupby('Academic Pressure_Category')['Depression'].agg(['mean', 'count']).iterrows():
        print(f"   - {level}: {row['mean']:.2%} (樣本數: {row['count']})")
    
    print("\n4. 研究建議:")
    print("   - 根據分析結果，應該提供更多的心理健康資源給高學業壓力學生")
    print("   - 學校可以發展壓力管理培訓課程，特別針對高風險學生群體")
    print("   - 進一步研究可以探索學業壓力與其他因素（如社交支持、睡眠質量）的交互作用")
    
    return results

def main():
    """
    主程式流程
    """
    # 解析命令列參數
    args = parse_args()
    
    # 設定環境與字型
    zh_font = setup_environment()
    
    # 讀取並前處理資料
    print("讀取並前處理資料...")
    try:
        df = preprocess(args.data_path)
    except Exception as e:
        print(f"資料處理失敗: {e}")
        sys.exit(1)
    
    # 執行基本資料分析
    run_basic_analysis(df, zh_font)
    
    # 執行模型訓練與評估
    model_results = run_model_analysis(df, zh_font)
    
    # 匯出資料至 MySQL（如果指定）
    if args.to_mysql:
        export_to_db(df)
    else:
        print("\n若要匯出資料至 MySQL，請使用 --to-mysql 參數")
    
    print("\n分析完成！")

# 當直接執行此模組時
if __name__ == "__main__":
    # 防止 plt.show() 阻塞程式執行
    plt.ion()
    
    # 執行主程式
    main()
    
    # 等待使用者按鍵結束程式
    plt.ioff()
    input("\n按下 Enter 鍵結束程式...")
