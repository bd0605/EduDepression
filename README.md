# 學生憂鬱症風險分析專案

版本：v1.0.0  
更新日期：2025-05-11

## 專案簡介

這是一個使用 XAMPP + MySQL + Grafana 進行學生憂鬱症風險分析的期末報告專案。透過機器學習方法（K-means 聚類和邏輯回歸）分析不同學歷層級學生的憂鬱症風險。

## 技術架構

- **資料庫**：MySQL (XAMPP)
- **程式語言**：Python 3.8+
- **視覺化**：Matplotlib, Seaborn, Grafana
- **機器學習**：Scikit-learn

## 專案結構

```
EduDepression/
├── data/                    # 資料集
├── db/                      # 資料庫設定
├── notebooks/               # Jupyter筆記本
├── report/                  # 分析報告
├── src/                     # 原始碼
├── visuals/                 # 圖表輸出
├── requirements.txt         # Python套件需求
├── run_analysis.py          # 主要分析程式
└── README.md                # 專案說明
```

## 快速開始

1. 安裝 XAMPP 和 Python 環境

2. 初始化資料庫（有兩種方式）：

### ✅ 方法一（推薦）：使用 phpMyAdmin（圖形化介面）

```
1. 開啟 http://localhost/phpmyadmin
2. 建立名為 student_depression 的資料庫
3. 點擊匯入 → 選擇 db/init_student_depression.sql → 執行
```

### 🧪 方法二：使用 MySQL CLI（進階用戶）

```bash
mysql -u root -p student_depression < db/init_student_depression.sql
```

3. 安裝 Python 套件：

```bash
pip install -r requirements.txt
```

4. 執行分析：

```bash
python run_analysis.py
```

## 主要功能

1. **資料處理**：載入學生憂鬱症資料集
2. **基礎統計**：計算不同學歷的憂鬱症比例
3. **K-means 聚類**：將學生分成 3 個風險群組
4. **邏輯回歸**：建立憂鬱症預測模型
5. **視覺化**：生成混淆矩陣、ROC 曲線等圖表
6. **Grafana 整合**：即時資料視覺化儀表板

## 待填寫項目

報告中預留了數值填寫區域，包括：

- 資料筆數和欄位數
- 各學歷憂鬱症比例
- K-means 聚類結果
- 邏輯回歸模型指標

## 作者

陳貝蒂 組長  
大學生期末小報告專案
