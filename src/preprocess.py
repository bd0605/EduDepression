# -*- coding: utf-8 -*-
"""
資料前處理與特徵工程模組

此模組負責 EduDepression 專案的資料讀取、清洗與特徵工程，
包含缺失值填補、變數轉換、離群值處理與分類變數處理等。
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    讀取原始 CSV 資料並處理基本映射關係

    Args:
        path (str): CSV 資料集路徑

    Returns:
        pandas.DataFrame: 載入後的原始資料集
    """
    # 讀取 CSV 檔案
    raw_df = pd.read_csv(path)
    print(f"原始資料集大小: {raw_df.shape}")
    print(f"資料集列名: {raw_df.columns.tolist()}")

    # 將憂鬱症類別轉換為數值型 (0/1)
    if raw_df['Depression'].dtype == object:
        raw_df['Depression'] = raw_df['Depression'].map(
            {'No': 0, 'Yes': 1}).astype(int)

    # 顯示基本資料統計
    print("\n基本資料統計 (數值型變數):")
    print(raw_df[['Age', 'Academic Pressure', 'Work Pressure',
          'CGPA', 'Study Satisfaction']].describe())

    # 確保學業壓力變數存在
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

    return raw_df

def process_degree(df):
    """
    處理學位變數，將各種學位描述轉換為四級分類

    Args:
        df (pandas.DataFrame): 包含 Degree 欄位的資料框

    Returns:
        pandas.DataFrame: 加入 Degree4 欄位的資料框
    """
    # 複製一份資料以避免修改原始資料
    result_df = df.copy()
    
    # 學位處理與四級分類
    deg_mode = result_df['Degree'].mode().iloc[0]
    result_df['Degree'] = result_df['Degree'].fillna(deg_mode).astype(str).str.strip()
    result_df.loc[result_df['Degree'] == '其他', 'Degree'] = deg_mode

    # 定義學位簡化函數
    def simplify_degree(x):
        x = x.lower()
        if 'phd' in x or '博士' in x:
            return '博士'
        if 'master' in x or 'msc' in x or '碩士' in x:
            return '碩士'
        if any(k in x for k in ['bachelor', 'ba', 'b.sc', 'bsc', 'bcom', 'be', 'mba']):
            return '大學'
        return '高中及以下'

    # 套用學位簡化函數
    result_df['Degree4'] = result_df['Degree'].apply(simplify_degree)
    
    return result_df

def engineer_features(df):
    """
    執行特徵工程，包含學業壓力分類、刪除重複/缺失值、處理離群值等

    Args:
        df (pandas.DataFrame): 原始資料框

    Returns:
        pandas.DataFrame: 處理後的資料框，含有增強特徵
    """
    # 複製一份資料
    result_df = df.copy().reset_index(drop=True)
    
    # 學業壓力分類
    if 'Academic Pressure' in result_df.columns:
        if result_df['Academic Pressure'].dtype == object:
            # 如果是類別資料，轉換為數值
            ap_values = {k: i+1 for i,
                        k in enumerate(sorted(result_df['Academic Pressure'].unique()))}
            result_df['Academic Pressure_Value'] = result_df['Academic Pressure'].map(ap_values)
            # 保留原始類別以便分析
            result_df['Academic Pressure_Category'] = result_df['Academic Pressure']
        else:
            # 如果已經是數值，進行分組
            result_df['Academic Pressure_Value'] = result_df['Academic Pressure']
            # 檢查數據的分佈
            ap_min = result_df['Academic Pressure_Value'].min()
            ap_max = result_df['Academic Pressure_Value'].max()

            # 使用 cut 進行分組，確保每個組有數據
            bins = [ap_min, ap_min + (ap_max-ap_min)/3,
                    ap_min + 2*(ap_max-ap_min)/3, ap_max]
            result_df['Academic Pressure_Category'] = pd.cut(
                result_df['Academic Pressure_Value'],
                bins=bins,
                labels=['低壓力', '中壓力', '高壓力'],
                include_lowest=True
            )
    else:
        print("警告: 資料集中找不到 Academic Pressure 欄位!")
        # 創建一個替代欄位
        result_df['Academic Pressure_Value'] = result_df['Work Pressure']  # 假設工作壓力可作為替代
        result_df['Academic Pressure_Category'] = pd.cut(
            result_df['Academic Pressure_Value'],
            bins=3,
            labels=['低壓力', '中壓力', '高壓力']
        )
    
    # 更嚴謹的數據清理
    result_df = result_df.drop_duplicates().dropna(
        subset=['Academic Pressure_Value', 'Depression']).reset_index(drop=True)

    # 數值特徵處理
    num_cols = ['Age', 'Academic Pressure_Value',
                'Work Pressure', 'CGPA', 'Study Satisfaction']
    result_df[num_cols] = result_df[num_cols].fillna(result_df[num_cols].median())
    
    # 移除離群值 (使用 Z-score)
    z = stats.zscore(result_df[num_cols])
    result_df = result_df[(np.abs(z) < 3).all(axis=1)].reset_index(drop=True)
    
    # 學歷序數與增強特徵
    order4 = ['高中及以下', '大學', '碩士', '博士']
    result_df['degree_ord4'] = result_df['Degree4'].map({d: i+1 for i, d in enumerate(order4)})
    
    # 性別 One-hot 編碼
    result_df = pd.get_dummies(result_df, columns=['Gender'], drop_first=True)
    
    return result_df

def scale_features(df, features):
    """
    標準化數值特徵

    Args:
        df (pandas.DataFrame): 包含待標準化特徵的資料框
        features (list): 需要標準化的特徵名稱列表

    Returns:
        tuple: (pandas.DataFrame, sklearn.preprocessing.StandardScaler)
            - 包含標準化後特徵的資料框
            - 訓練完畢的 StandardScaler 物件，可用於後續轉換
    """
    # 初始化標準化器
    scaler = StandardScaler()
    
    # 轉換資料
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features,
        index=df.index
    )
    
    return df_scaled, scaler

def preprocess(path):
    """
    整合資料預處理流程

    Args:
        path (str): 原始資料路徑

    Returns:
        pandas.DataFrame: 完整預處理後的資料框
    """
    # 載入資料
    raw_df = load_data(path)
    
    # 處理學位資料
    processed_df = process_degree(raw_df)
    
    # 進行特徵工程
    result_df = engineer_features(processed_df)
    
    return result_df

# 當直接執行此模組時進行測試
if __name__ == "__main__":
    # 假設資料在標準路徑
    data_path = "data/student_depression_dataset.csv"
    
    # 測試資料讀取
    try:
        df = preprocess(data_path)
        print(f"成功處理資料！資料形狀: {df.shape}")
        print(f"欄位: {df.columns.tolist()}")
    except Exception as e:
        print(f"資料處理發生錯誤: {e}")
