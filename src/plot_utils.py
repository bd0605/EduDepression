# -*- coding: utf-8 -*-
"""
核心視覺化函式模組

此模組提供 EduDepression 專案的所有視覺化功能，包含學業壓力與憂鬱風險的
條形圖、分布圖、ROC 曲線、特徵重要性等視覺化工具。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve

def setup_chinese_font(font_path=None):
    """
    設定中文字型，用於圖表中顯示中文

    Args:
        font_path (str, optional): 中文字型檔案路徑，若為 None 則使用預設路徑

    Returns:
        matplotlib.font_manager.FontProperties: 中文字型屬性物件
    """
    if font_path is None:
        # 預設路徑，依系統環境可能需調整
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    
    # 載入字型
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style('whitegrid')
    
    # 回傳字型屬性物件，供後續圖表使用
    return FontProperties(fname=font_path)

def plot_depression_by_pressure_level(df, zh_font, figsize=(8, 6), save_path=None):
    """
    繪製不同學業壓力水平的憂鬱比例條形圖

    Args:
        df (pandas.DataFrame): 包含 Academic Pressure_Category 與 Depression 欄位的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        figsize (tuple, optional): 圖表尺寸 (width, height)，預設為 (8, 6)
        save_path (str, optional): 圖片儲存路徑，若為 None 則不儲存

    Returns:
        matplotlib.figure.Figure: 繪製完成的圖表物件
    """
    # 建立圖表
    plt.figure(figsize=figsize)
    
    # 計算各壓力組的憂鬱比例
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

    # 繪製條形圖
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

    # 設定標題與軸標籤
    plt.xticks(fontproperties=zh_font, fontsize=14)
    plt.yticks(fontproperties=zh_font, fontsize=14)
    plt.xlabel('學業壓力水平', fontproperties=zh_font)
    plt.ylabel('憂鬱比例', fontproperties=zh_font)
    plt.title('不同學業壓力水平的憂鬱比例', fontproperties=zh_font)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 若有指定儲存路徑，則儲存圖片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.show()
    
    return plt.gcf()

def plot_pressure_bins_distribution(df, zh_font, bins=5, figsize=(8, 6), save_path=None):
    """
    繪製學業壓力連續值分箱後的憂鬱比例分布

    Args:
        df (pandas.DataFrame): 包含 Academic Pressure_Value 與 Depression 欄位的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        bins (int, optional): 分箱數量，預設為 5
        figsize (tuple, optional): 圖表尺寸 (width, height)，預設為 (8, 6)
        save_path (str, optional): 圖片儲存路徑，若為 None 則不儲存

    Returns:
        matplotlib.figure.Figure: 繪製完成的圖表物件
    """
    # 建立圖表
    plt.figure(figsize=figsize)
    
    # 創建分箱
    n_bins = bins
    min_val = df['Academic Pressure_Value'].min()
    max_val = df['Academic Pressure_Value'].max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    # 創建分組標籤
    bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(n_bins)]

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

        # 設定標題與軸標籤
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
    
    # 若有指定儲存路徑，則儲存圖片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.show()
    
    return plt.gcf()

def plot_feature_importance_bar(importances, zh_font, title, top_n=5, figsize=(8, 6), save_path=None):
    """
    繪製特徵重要性條形圖

    Args:
        importances (pandas.Series): 包含特徵名稱與重要性分數的 Series
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        title (str): 圖表標題
        top_n (int, optional): 顯示前幾個重要特徵，預設為 5
        figsize (tuple, optional): 圖表尺寸 (width, height)，預設為 (8, 6)
        save_path (str, optional): 圖片儲存路徑，若為 None 則不儲存

    Returns:
        matplotlib.figure.Figure: 繪製完成的圖表物件
    """
    # 建立圖表
    plt.figure(figsize=figsize)
    
    # 選取前 N 個特徵
    top_features = importances.sort_values(ascending=False).head(top_n)
    
    # 繪製條形圖
    top_features.plot(kind='bar')
    
    # 設定標題與軸標籤
    plt.title(title, fontproperties=zh_font, fontsize=14)
    plt.xlabel('特徵', fontproperties=zh_font, fontsize=12)
    plt.ylabel('重要性分數', fontproperties=zh_font, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontproperties=zh_font)
    
    # 若有指定儲存路徑，則儲存圖片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def plot_confusion_matrix(cm, zh_font, title='混淆矩陣', figsize=(6, 5), save_path=None):
    """
    繪製混淆矩陣熱圖

    Args:
        cm (numpy.ndarray): 混淆矩陣，通常是 2x2 (二元分類) 或 n x n (多元分類)
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        title (str, optional): 圖表標題，預設為 '混淆矩陣'
        figsize (tuple, optional): 圖表尺寸 (width, height)，預設為 (6, 5)
        save_path (str, optional): 圖片儲存路徑，若為 None 則不儲存

    Returns:
        matplotlib.figure.Figure: 繪製完成的圖表物件
    """
    # 建立圖表
    plt.figure(figsize=figsize)
    
    # 繪製混淆矩陣熱圖
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['預測 0（無憂鬱）', '預測 1（有憂鬱）'],
        yticklabels=['實際 0（無憂鬱）', '實際 1（有憂鬱）'],
        annot_kws={'fontproperties': zh_font, 'fontsize': 11}
    )
    
    # 設定標題與軸標籤
    plt.title(title, fontproperties=zh_font, fontsize=14)
    plt.ylabel('實際標籤', fontproperties=zh_font, fontsize=12)
    plt.xlabel('預測標籤', fontproperties=zh_font, fontsize=12)
    plt.xticks(fontproperties=zh_font, rotation=0)
    plt.yticks(fontproperties=zh_font, rotation=0)
    
    # 若有指定儲存路徑，則儲存圖片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def plot_roc_curves(roc_data, zh_font, figsize=(8, 6), save_path=None):
    """
    繪製一個或多個模型的 ROC 曲線

    Args:
        roc_data (list): 包含 (y_test, y_prob, label, auc) 四元組的列表，每個四元組對應一個模型
            - y_test (array-like): 實際標籤
            - y_prob (array-like): 預測機率
            - label (str): 模型名稱
            - auc (float): AUC 分數
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        figsize (tuple, optional): 圖表尺寸 (width, height)，預設為 (8, 6)
        save_path (str, optional): 圖片儲存路徑，若為 None 則不儲存

    Returns:
        matplotlib.figure.Figure: 繪製完成的圖表物件
    """
    # 建立圖表
    plt.figure(figsize=figsize)
    
    # 為每個模型繪製 ROC 曲線
    for y_test, y_prob, label, auc in roc_data:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})')
    
    # 繪製對角線 (隨機猜測基準線)
    plt.plot([0, 1], [0, 1], 'k--')
    
    # 設定軸範圍
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # 設定標題與軸標籤
    plt.xlabel('偽陽性率 (FPR)', fontproperties=zh_font, fontsize=12)
    plt.ylabel('真陽性率 (TPR)', fontproperties=zh_font, fontsize=12)
    plt.title('預測憂鬱風險的 ROC 曲線比較', fontproperties=zh_font, fontsize=14)
    plt.legend(loc="lower right", prop=zh_font)
    plt.grid(True)
    
    # 若有指定儲存路徑，則儲存圖片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.show()
    
    return plt.gcf()

def plot_combined_depression_charts(df, zh_font, figsize=(16, 6), save_path=None):
    """
    在單一圖表中同時繪製壓力分類與壓力連續值的憂鬱比例

    Args:
        df (pandas.DataFrame): 包含 Academic Pressure_Category、Academic Pressure_Value 與 Depression 欄位的資料框
        zh_font (matplotlib.font_manager.FontProperties): 中文字型屬性物件
        figsize (tuple, optional): 圖表尺寸 (width, height)，預設為 (16, 6)
        save_path (str, optional): 圖片儲存路徑，若為 None 則不儲存

    Returns:
        matplotlib.figure.Figure: 繪製完成的圖表物件
    """
    # 建立圖表
    plt.figure(figsize=figsize)
    
    # 第一個子圖：壓力分類憂鬱比例
    plt.subplot(1, 2, 1)
    
    # 計算各壓力組的憂鬱比例
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

    # 繪製條形圖
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

    # 設定標題與軸標籤
    plt.xticks(fontproperties=zh_font, fontsize=14)
    plt.yticks(fontproperties=zh_font, fontsize=14)
    plt.xlabel('學業壓力水平', fontproperties=zh_font)
    plt.ylabel('憂鬱比例', fontproperties=zh_font)
    plt.title('不同學業壓力水平的憂鬱比例', fontproperties=zh_font)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 第二個子圖：壓力連續值分箱憂鬱比例
    plt.subplot(1, 2, 2)
    
    # 創建分箱
    n_bins = 5
    min_val = df['Academic Pressure_Value'].min()
    max_val = df['Academic Pressure_Value'].max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    # 創建分組標籤
    bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(n_bins)]

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

        # 設定標題與軸標籤
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
    
    # 調整子圖間距
    plt.tight_layout()
    
    # 若有指定儲存路徑，則儲存圖片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.show()
    
    return plt.gcf()

# 當直接執行此模組時進行測試
if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess
    
    # 假設資料在標準路徑
    data_path = "data/student_depression_dataset.csv"
    
    try:
        # 讀取並處理資料
        df = preprocess(data_path)
        
        # 設定中文字型
        zh_font = setup_chinese_font()
        
        # 測試視覺化函式
        plot_depression_by_pressure_level(df, zh_font)
        plot_pressure_bins_distribution(df, zh_font)
        
        print("視覺化函式測試成功！")
    except Exception as e:
        print(f"視覺化函式測試失敗: {e}")
