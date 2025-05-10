# 快速開始指南

版本：v1.0.0  
更新日期：2025-05-11

## 1. 環境準備

### 安裝 XAMPP

1. 下載 XAMPP：https://www.apachefriends.org/
2. 安裝並啟動 Apache 和 MySQL 服務

### Python 環境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 資料庫設定

1. 開啟 phpMyAdmin（http://localhost/phpmyadmin）
2. 建立資料庫：
   ```sql
   CREATE DATABASE student_depression;
   ```
3. 執行初始化腳本：
   ```sql
   source db/init_student_depression.sql
   ```

## 3. 執行分析

```bash
python run_analysis.py
```

執行後會：

- 載入資料到 MySQL
- 產生統計分析結果
- 建立視覺化圖表（visuals/目錄）
- 生成報告模板（report/目錄）

## 4. 設定 Grafana（選用）

1. 安裝 Grafana
2. 新增 MySQL 資料源
3. 匯入儀表板設定

## 5. 填寫報告

開啟 `report/analysis_report.md`，根據執行結果填寫預留的數值區域。

## 常見問題

1. **MySQL 連接失敗**：確認 XAMPP 已啟動 MySQL 服務
2. **套件安裝失敗**：確認使用 Python 3.8+
3. **資料載入錯誤**：確認資料集位於 data/ 目錄
