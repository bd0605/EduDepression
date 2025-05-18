# -*- coding: utf-8 -*-
"""
模型訓練與評估模組

此模組提供 EduDepression 專案的機器學習模型訓練、評估與特徵重要性分析功能，
支援 Logistic Regression 與 Random Forest 模型，以及交叉驗證、ROC 曲線分析等。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.inspection import permutation_importance

def prepare_features(df, features=None, target='Depression', test_size=0.2, random_state=42):
    """
    準備模型訓練與測試的特徵集與標籤集

    Args:
        df (pandas.DataFrame): 資料框
        features (list, optional): 特徵名稱列表，若為 None 則自動選擇數值欄位
        target (str, optional): 目標變數名稱，預設為 'Depression'
        test_size (float, optional): 測試集比例，預設為 0.2
        random_state (int, optional): 隨機種子，預設為 42

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
            - X_train: 訓練特徵集
            - X_test: 測試特徵集
            - y_train: 訓練標籤集
            - y_test: 測試標籤集
            - scaler: 標準化器物件
    """
    # 若未指定特徵，則選擇所有數值欄位
    if features is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in num_cols if col != target]
    
    # 準備特徵與標籤
    X = df[features]
    y = df[target]
    
    # 標準化特徵
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

def train_logistic_regression(X_train, y_train, class_weight='balanced', solver='liblinear',
                           max_iter=5000, random_state=42):
    """
    訓練 Logistic Regression 模型

    Args:
        X_train (array-like): 訓練特徵集
        y_train (array-like): 訓練標籤集
        class_weight (str or dict, optional): 類別權重設定，預設為 'balanced'
        solver (str, optional): 最佳化演算法，預設為 'liblinear'
        max_iter (int, optional): 最大迭代次數，預設為 5000
        random_state (int, optional): 隨機種子，預設為 42

    Returns:
        sklearn.linear_model.LogisticRegression: 訓練完成的模型
    """
    # 建立並訓練模型
    model = LogisticRegression(
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None,
                     class_weight='balanced', random_state=42, n_jobs=-1):
    """
    訓練 Random Forest 模型

    Args:
        X_train (array-like): 訓練特徵集
        y_train (array-like): 訓練標籤集
        n_estimators (int, optional): 決策樹數量，預設為 100
        max_depth (int, optional): 決策樹最大深度，預設為 None (無限制)
        class_weight (str or dict, optional): 類別權重設定，預設為 'balanced'
        random_state (int, optional): 隨機種子，預設為 42
        n_jobs (int, optional): 平行處理的工作數，預設為 -1 (使用所有核心)

    Returns:
        sklearn.ensemble.RandomForestClassifier: 訓練完成的模型
    """
    # 建立並訓練模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    評估模型效能

    Args:
        model (object): 訓練完成的模型
        X_test (array-like): 測試特徵集
        y_test (array-like): 測試標籤集
        model_name (str, optional): 模型名稱，用於輸出，預設為 "Model"

    Returns:
        dict: 包含各種效能指標的字典
            - y_pred: 預測標籤
            - y_proba: 預測機率
            - accuracy: 準確率
            - auc: AUC 分數
            - confusion_matrix: 混淆矩陣
            - classification_report: 分類報告
    """
    # 預測結果
    y_pred = model.predict(X_test)
    
    # 嘗試取得預測機率，不同模型有不同方式
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = y_pred  # 若無法獲得機率，則使用預測標籤
    
    # 計算效能指標
    acc = accuracy_score(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = None  # 若無法計算 AUC，則設為 None
    
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    # 輸出效能摘要
    print(f"\n{model_name} 評估結果:")
    print(f"準確率: {acc:.3f}")
    if auc is not None:
        print(f"AUC: {auc:.3f}")
    
    # 返回效能指標
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm,
        "classification_report": cr
    }

def get_feature_importance(model, X, y, features, model_type="logistic_regression"):
    """
    取得特徵重要性

    Args:
        model (object): 訓練完成的模型
        X (array-like): 特徵資料
        y (array-like): 標籤資料
        features (list): 特徵名稱列表
        model_type (str, optional): 模型類型，可選 "logistic_regression" 或 "random_forest"，預設為 "logistic_regression"

    Returns:
        pandas.Series: 特徵重要性分數，索引為特徵名稱
    """
    if model_type == "logistic_regression":
        # Logistic Regression 的特徵重要性為係數絕對值
        try:
            importance = pd.Series(np.abs(model.coef_[0]), index=features)
        except:
            importance = pd.Series(np.abs(model.coef_.ravel()), index=features)
    elif model_type == "random_forest":
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        importance = pd.Series(result.importances_mean, index=features)
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")
    
    # 排序並返回
    return importance.sort_values(ascending=False)

def cross_validate_models(X, y, features, cv=5, random_state=42):
    """
    使用交叉驗證比較不同模型

    Args:
        X (array-like): 特徵資料
        y (array-like): 標籤資料
        features (list): 特徵名稱列表
        cv (int, optional): 交叉驗證的折數，預設為 5
        random_state (int, optional): 隨機種子，預設為 42

    Returns:
        dict: 各模型的交叉驗證結果
            - logistic_regression: Logistic Regression 的交叉驗證分數
            - random_forest: Random Forest 的交叉驗證分數
    """
    # 建立模型
    lr = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=5000, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)
    
    # 進行交叉驗證
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    
    # 輸出結果
    print("\n交叉驗證結果:")
    print(f"Logistic Regression: {lr_scores.mean():.3f} (+/- {lr_scores.std()*2:.3f})")
    print(f"Random Forest: {rf_scores.mean():.3f} (+/- {rf_scores.std()*2:.3f})")
    
    # 返回結果
    return {
        "logistic_regression": lr_scores,
        "random_forest": rf_scores
    }

def check_correlation_with_depression(df, target='Depression'):
    """
    檢查各特徵與憂鬱風險的相關性

    Args:
        df (pandas.DataFrame): 資料框
        target (str, optional): 目標變數名稱，預設為 'Depression'

    Returns:
        pandas.Series: 特徵與目標變數的相關係數，已排序
    """
    # 選擇數值型欄位
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 計算相關係數
    corr = df[num_cols].corrwith(df[target]).drop(target, errors='ignore')
    
    # 排序並返回
    return corr.sort_values(ascending=False)

def train_and_evaluate(df, features=None, target='Depression', test_size=0.2, random_state=42):
    """
    整合式模型訓練與評估流程

    Args:
        df (pandas.DataFrame): 資料框
        features (list, optional): 特徵名稱列表，若為 None 則自動選擇數值欄位
        target (str, optional): 目標變數名稱，預設為 'Depression'
        test_size (float, optional): 測試集比例，預設為 0.2
        random_state (int, optional): 隨機種子，預設為 42

    Returns:
        dict: 模型評估結果
            - lr_model: Logistic Regression 模型
            - lr_results: Logistic Regression 評估結果
            - rf_model: Random Forest 模型
            - rf_results: Random Forest 評估結果
            - lr_importance: Logistic Regression 特徵重要性
            - rf_importance: Random Forest 特徵重要性
    """
    # 準備資料
    if features is None:
        # 若未指定特徵，則選擇數值欄位與部分特殊欄位
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in num_cols if col != target]
        # 加入 degree_ord4 與 Gender 欄位（如果存在）
        for col in ['degree_ord4']:
            if col in df.columns and col not in features:
                features.append(col)
        # 加入所有 Gender_ 開頭的 one-hot 欄位
        gender_cols = [col for col in df.columns if col.startswith('Gender_')]
        for col in gender_cols:
            if col not in features:
                features.append(col)
    
    # 檢查相關性
    corr_with_target = check_correlation_with_depression(df, target)
    print("\n各特徵與憂鬱風險的相關性:")
    print(corr_with_target.head(10))
    
    # 分割資料集並標準化
    X_train, X_test, y_train, y_test, scaler = prepare_features(
        df, features, target, test_size, random_state
    )
    
    # 訓練 Logistic Regression 模型
    lr_model = train_logistic_regression(X_train, y_train, random_state=random_state)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 訓練 Random Forest 模型
    rf_model = train_random_forest(X_train, y_train, random_state=random_state)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # 計算特徵重要性
    lr_importance = get_feature_importance(
        lr_model, X_test, y_test, features, "logistic_regression"
    )
    print("\nLogistic Regression 特徵重要性 (Top 5):")
    print(lr_importance.head(5))
    
    rf_importance = get_feature_importance(
        rf_model, X_test, y_test, features, "random_forest"
    )
    print("\nRandom Forest 特徵重要性 (Top 5):")
    print(rf_importance.head(5))
    
    # 檢查學業壓力在特徵重要性中的排名
    ap_lr_rank = -1
    ap_rf_rank = -1
    
    if 'Academic Pressure_Value' in features:
        ap_lr_rank = list(lr_importance.index).index('Academic Pressure_Value') + 1
        ap_rf_rank = list(rf_importance.index).index('Academic Pressure_Value') + 1
        
        print(f"\n學業壓力在模型中的重要性排名:")
        print(f"- Logistic Regression: 第 {ap_lr_rank} 名")
        print(f"- Random Forest: 第 {ap_rf_rank} 名")
    
    # 返回結果
    return {
        "lr_model": lr_model,
        "lr_results": lr_results,
        "rf_model": rf_model,
        "rf_results": rf_results,
        "lr_importance": lr_importance,
        "rf_importance": rf_importance,
        "ap_lr_rank": ap_lr_rank,
        "ap_rf_rank": ap_rf_rank,
        "features": features,
        "scaler": scaler,
        "y_train": y_train,
        "y_test":  y_test
    }

# 當直接執行此模組時進行測試
if __name__ == "__main__":
    import os
    from preprocess import preprocess
    
    # 假設資料在標準路徑
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "student_depression_dataset.csv")
    
    try:
        # 讀取並處理資料
        df = preprocess(data_path)
        
        # 指定特徵
        features = [
            'Academic Pressure_Value', 'degree_ord4', 'Age', 
            'CGPA', 'Study Satisfaction'
        ]
        features.extend([col for col in df.columns if col.startswith('Gender_')])
        
        # 訓練與評估模型
        results = train_and_evaluate(df, features)
        
        print("\n模型訓練與評估成功！")
        print(f"Logistic Regression 準確率: {results['lr_results']['accuracy']:.3f}, AUC: {results['lr_results']['auc']:.3f}")
        print(f"Random Forest 準確率: {results['rf_results']['accuracy']:.3f}, AUC: {results['rf_results']['auc']:.3f}")
        
        # 交叉驗證比較
        X = df[features]
        y = df['Depression']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cross_validate_models(X_scaled, y, features)
        
    except Exception as e:
        print(f"模型訓練與評估失敗: {e}")
