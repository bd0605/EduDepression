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
import time
from tqdm import tqdm

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

class AnalysisProgressTracker:
    """統一的分析進度追蹤器"""
    
    def __init__(self, export_to_mysql=False):
        # 根據是否匯出MySQL決定總步驟數
        self.total_steps = 5 if export_to_mysql else 4
        self.current_step = 0
        self.pbar = None
        
        # 步驟定義
        self.steps = [
            ("🔧 環境設定", "設定字型與執行環境"),
            ("📊 資料預處理", "載入與清理資料"),
            ("📈 統計分析", "計算相關性與統計檢定"),
            ("🤖 模型訓練", "機器學習模型訓練與評估")
        ]
        
        if export_to_mysql:
            self.steps.append(("🗄️ 資料匯出", "匯出資料至MySQL資料庫"))
    
    def start(self):
        """開始進度追蹤"""
        print("🚀 EduDepression 學業壓力與憂鬱風險分析系統")
        print("=" * 60)
        self.pbar = tqdm(
            total=100, 
            desc="準備開始", 
            unit="%", 
            ncols=80,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}'
        )
    
    def next_step(self, detail=""):
        """進入下一個步驟"""
        if self.pbar is None:
            return
            
        self.current_step += 1
        step_name, step_desc = self.steps[self.current_step - 1]
        
        # 計算進度百分比
        progress = int((self.current_step - 1) / self.total_steps * 100)
        self.pbar.n = progress
        
        # 更新描述
        desc = f"{step_name}"
        if detail:
            desc += f" - {detail}"
        self.pbar.set_description(desc)
        self.pbar.refresh()
        
        # 在控制台顯示步驟資訊
        print(f"\n{step_name}: {step_desc}")
        if detail:
            print(f"  → {detail}")
    
    def update_detail(self, detail):
        """更新當前步驟的詳細資訊"""
        if self.pbar is None:
            return
            
        step_name, _ = self.steps[self.current_step - 1]
        desc = f"{step_name} - {detail}"
        self.pbar.set_description(desc)
        self.pbar.refresh()
    
    def finish_step(self):
        """完成當前步驟"""
        if self.pbar is None:
            return
            
        # 更新到下一個步驟的起始點
        progress = int(self.current_step / self.total_steps * 100)
        self.pbar.n = progress
        self.pbar.refresh()
    
    def complete(self):
        """完成所有分析"""
        if self.pbar is None:
            return
            
        self.pbar.n = 100
        self.pbar.set_description("✅ 分析完成")
        self.pbar.refresh()
        self.pbar.close()

def setup_environment(progress_tracker):
    """
    設定執行環境，包含字型設定與警告過濾等

    Args:
        progress_tracker: 進度追蹤器
        
    Returns:
        matplotlib.font_manager.FontProperties: 中文字型屬性物件
    """
    progress_tracker.next_step("下載與設定中文字型")
    
    # 下載字型
    progress_tracker.update_detail("下載中文字型檔案")
    font_path = download_font_if_not_exist()
    
    # 載入字型
    progress_tracker.update_detail("載入字型至matplotlib")
    fm.fontManager.addfont(font_path)
    
    # 設定 matplotlib
    progress_tracker.update_detail("設定圖表顯示參數")
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus'] = False
    
    progress_tracker.finish_step()
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

def run_data_preprocessing(args, progress_tracker):
    """
    執行資料預處理

    Args:
        args: 命令列參數
        progress_tracker: 進度追蹤器
        
    Returns:
        pandas.DataFrame: 處理後的資料框
    """
    progress_tracker.next_step("載入原始資料集")
    
    try:
        progress_tracker.update_detail("讀取CSV檔案")
        df = preprocess(args.data_path)
        
        progress_tracker.update_detail(f"資料預處理完成 ({len(df)} 筆)")
        print(f"  處理後資料集大小: {df.shape}")
        
        progress_tracker.finish_step()
        return df
        
    except Exception as e:
        print(f"  ❌ 資料處理失敗: {e}")
        sys.exit(1)

def run_statistical_analysis(df, zh_font, progress_tracker):
    """
    執行統計分析

    Args:
        df (pandas.DataFrame): 處理後的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        progress_tracker: 進度追蹤器
    """
    progress_tracker.next_step("計算相關係數與統計檢定")
    
    # 計算學業壓力與憂鬱風險的相關係數
    progress_tracker.update_detail("計算壓力與憂鬱的相關性")
    ap_corr = df['Academic Pressure_Value'].corr(df['Depression'])
    print(f"  學業壓力與憂鬱風險的相關係數: {ap_corr:.3f}")

    # 分析各壓力組的憂鬱比例
    progress_tracker.update_detail("分析各壓力組憂鬱比例")
    ap_group = df.groupby('Academic Pressure_Category')[
        'Depression'].agg(['mean', 'count'])
    ap_group.columns = ['憂鬱比例', '樣本數']
    print("\n  不同學業壓力水平的憂鬱風險:")
    print(ap_group.to_string(index=True))
    
    # 顯示各壓力組詳細統計
    for category in ['低壓力', '中壓力', '高壓力']:
        if category in df['Academic Pressure_Category'].values:
            subset = df[df['Academic Pressure_Category'] == category]
            depression_rate = subset['Depression'].mean()
            count = len(subset)
            print(f"    {category}: 憂鬱比例 = {depression_rate:.1%}, 樣本數 = {count}")
    
    # 創建交叉列聯表與卡方檢定
    progress_tracker.update_detail("執行卡方獨立性檢定")
    contingency_table = pd.crosstab(
        df['Academic Pressure_Category'],
        df['Depression']
    )
    print(f"\n  交叉列聯表：")
    print(contingency_table.to_string())
    
    # 執行卡方檢定
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\n  卡方獨立性檢定結果:")
    print(f"    卡方值: {chi2:.3f}, 自由度: {dof}, p-value: {p_value:.4f}")
    print(f"    結論: {'壓力等級之間憂鬱風險有顯著差異' if p_value < 0.05 else '壓力等級之間憂鬱風險沒有顯著差異'}")
    
    # 繪製視覺化圖表
    progress_tracker.update_detail("生成統計視覺化圖表")
    plot_combined_depression_charts(df, zh_font)
    
    progress_tracker.finish_step()

def run_model_analysis(df, zh_font, progress_tracker):
    """
    執行模型訓練與評估

    Args:
        df (pandas.DataFrame): 處理後的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        progress_tracker: 進度追蹤器
        
    Returns:
        dict: 模型訓練與評估結果
    """
    progress_tracker.next_step("特徵選擇與模型訓練")
    
    # 選擇特徵
    progress_tracker.update_detail("選擇模型特徵變數")
    features = [
        'Academic Pressure_Value', 'degree_ord4', 'Age', 
        'CGPA', 'Study Satisfaction'
    ]
    features.extend([col for col in df.columns if col.startswith('Gender_')])
    
    # 訓練與評估模型
    progress_tracker.update_detail("訓練Logistic Regression和Random Forest模型")
    print("\n====== 預測模型建立與評估 ======")
    results = train_and_evaluate(df, features)

    # 繪製模型評估圖表
    progress_tracker.update_detail("生成模型評估圖表")
    
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
    
    progress_tracker.finish_step()
    
    # 輸出研究結論
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

def export_data_to_mysql(df, progress_tracker):
    """
    將資料匯出至 MySQL 資料庫

    Args:
        df (pandas.DataFrame): 處理後的資料框
        progress_tracker: 進度追蹤器
        
    Returns:
        bool: 匯出是否成功
    """
    progress_tracker.next_step("連接MySQL並匯出資料")
    
    # 測試資料庫連接
    progress_tracker.update_detail("測試MySQL資料庫連接")
    if not test_connection():
        print("  ❌ 無法連接至 MySQL 資料庫，請檢查連接設定")
        return False
    
    # 建立資料庫結構
    progress_tracker.update_detail("建立資料庫表格結構")
    sql_file_path = os.path.join(os.path.dirname(__file__), "db", "create_table.sql")
    if os.path.exists(sql_file_path):
        print("  建立資料庫結構...")
        create_schema(sql_file_path)
    
    # 匯出資料
    progress_tracker.update_detail("執行資料匯出作業")
    print("  匯出資料至 MySQL...")
    success = export_to_mysql(df, "student_depression")
    
    # 回報結果
    if success:
        print("  ✅ 資料成功匯出至 MySQL！")
        print("  💡 您現在可以使用 Grafana 連接 MySQL 進行視覺化")
    else:
        print("  ❌ 資料匯出失敗！")
    
    progress_tracker.finish_step()
    return success

def main():
    """
    主程式流程
    """
    # 解析命令列參數
    args = parse_args()
    
    # 初始化進度追蹤器
    progress_tracker = AnalysisProgressTracker(export_to_mysql=args.to_mysql)
    progress_tracker.start()
    
    try:
        # 設定環境與字型
        zh_font = setup_environment(progress_tracker)
        
        # 讀取並前處理資料
        df = run_data_preprocessing(args, progress_tracker)
        
        # 執行統計分析
        run_statistical_analysis(df, zh_font, progress_tracker)
        
        # 執行模型訓練與評估
        model_results = run_model_analysis(df, zh_font, progress_tracker)
        
        # 匯出資料至 MySQL（如果指定）
        if args.to_mysql:
            export_data_to_mysql(df, progress_tracker)
        else:
            print("\n💡 若要匯出資料至 MySQL，請使用 --to-mysql 參數")
        
        # 完成分析
        progress_tracker.complete()
        print("\n🎉 所有分析已成功完成！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷了分析程序")
        progress_tracker.complete()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 分析過程中發生錯誤: {e}")
        progress_tracker.complete()
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
