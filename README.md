# 學業壓力與憂鬱風險相關性分析

---

## 專案簡介

本專案以 27,901 筆大學生問卷資料為基礎，探討「學業壓力」（Academic Pressure）與「憂鬱風險」（Depression）之間的關聯性。  
採用資料前處理、分組比較、卡方檢定、主成分分析（PCA）與機器學習模型（Logistic Regression、Random Forest），並透過視覺化呈現結果。  
同時提供一套「可維護、易擴充」的 Python 分析框架，並整合 MySQL + Grafana 的資料視覺化方案。

---

## 專案結構

```
student-depression-analysis/
├── data/
│   └── student_depression_dataset.csv # kaggle 問卷資料
├── db/
│   └── create_table.sql               # MySQL 建表語法
├── docs/
│   ├── one-hot-gender-encoding.md     # 前處理補充文件
│   ├── 重構記錄.md                    # 重構過程紀錄
│   ├── mysql_grafana_guide.md         # MySQL與Grafana整合指南
│   ├── grafana_dashboard.json         # Grafana儀表板設定
│   └── 擴充概述.md                    # 專案擴充總結
├── fonts/                             # 字體文件目錄（自動下載）
├── report/
│   ├── 大學報告.md                    # 學術報告（Markdown）
│   └── 大學報告.docx                  # 學術報告（Word）
├── src/
│   ├── __init__.py
│   ├── preprocess.py                  # 資料前處理與特徵工程
│   ├── db_utils.py                    # MySQL 匯入匯出函式
│   ├── plot_utils.py                  # 核心視覺化函式
│   ├── model_utils.py                 # 模型訓練與評估
│   └── font_loader.py                 # 中文字體自動下載與管理
├── run_analysis.py                    # 主控程式：載入 → 處理 →（可選）DB → 繪圖
├── requirements.txt                   # Python 套件依賴
├── README.md                          # 本檔案
└── .gitignore                         # Git忽略檔案設定
```

---

## 技術架構

```text
┌────────────────────────────────────┐
│          使用者工作站             │
│  • Python 3.8+                     │
│  • pandas, numpy, scipy            │
│  • scikit-learn, seaborn           │
│  • matplotlib                      │
│  • sqlalchemy, pymysql             │
└────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│       XAMPP (MySQL Server)         │
│  • MySQL 8.0+ (port:3306)          │
│  • 學業壓力清洗後資料表            │
└────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│        Grafana 儀表板             │
│  • 連接 MySQL                      │
│  • 四種視覺化面板:                 │
│    - 壓力層級憂鬱比例條形圖        │
│    - 學歷vs壓力vs憂鬱線圖          │
│    - 性別統計表格                  │
│    - 壓力vs憂鬱熱力圖              │
└────────────────────────────────────┘
```

---

## 主要特性

- **模組化設計**：拆分前處理、視覺化、模型訓練與資料庫操作為獨立模組
- **跨平台相容**：自動字體管理，支援 Windows、macOS 與 Linux
- **完整資料庫整合**：支援 MySQL 資料匯出與 Grafana 視覺化整合
- **強大的視覺化**：條形圖、線圖、熱力圖與 ROC 曲線等多種視覺化方式
- **機器學習模型**：整合 Logistic Regression 與 Random Forest 模型預測
- **統計分析**：支援卡方檢定、主成分分析與相關性分析

---

## 快速開始

### 1. **克隆專案**

```bash
git clone https://github.com/bd0605/EduDepression.git
cd EduDepression
```

---

### 2. **建立虛擬環境與安裝依賴**

#### ✅ 推薦方式：使用 `uv`

```bash
# 若尚未安裝 uv，先執行：
pip install uv

# 建立虛擬環境
uv venv .venv

# 啟用虛擬環境
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

# 安裝依賴
uv pip install -r requirements.txt
```

#### 🧪 傳統方式（使用 pip）

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

### 3. **執行分析**

```bash
python run_analysis.py
```

- 預設：執行資料前處理、模型訓練、圖表分析
- 若需將結果匯入 MySQL，可加上參數：

```bash
python run_analysis.py --to-mysql
```

- 第一次運行時，程式會自動下載或尋找可用的中文字體，無需手動設定

---

## MySQL 與 Grafana 設定

### 1. 安裝 XAMPP

- 下載：[https://www.apachefriends.org/](https://www.apachefriends.org/)
- 安裝時勾選 **Apache**（選用）、**MySQL**（必要）

### 2. 啟動服務

1. 開啟 **XAMPP Control Panel**
2. 啟動 **MySQL**（必要）
3. 啟動 **Apache**（若需 phpMyAdmin / 靜態頁面）

### 3. 匯出資料至 MySQL

```bash
python run_analysis.py --to-mysql
```

系統會自動嘗試：

- 連接至本地 MySQL 伺服器
- 建立 `depression_db` 資料庫（若尚未存在）
- 執行 `db/create_table.sql` 建立表結構
- 匯入處理後的資料至 `student_depression` 資料表

### 4. 設定 Grafana

1. 安裝 Grafana：[官方安裝指南](https://grafana.com/docs/grafana/latest/installation/)
2. 啟動 Grafana 服務
3. 訪問 Grafana 網頁界面（預設 `http://localhost:3000/`）
4. 新增 MySQL 資料來源
5. 匯入 `docs/grafana_dashboard.json` 設定檔
6. 即可使用預先設計的視覺化儀表板

---

## 模組說明

### `src/preprocess.py`

資料前處理與特徵工程模組，包含：

- `load_data(path)`: 讀取 CSV 並轉換 Depression 為 0/1
- `process_degree(df)`: 將學位類別處理為四級分類（高中及以下/大學/碩士/博士）
- `engineer_features(df)`: 進行特徵工程，包含：
  - 學業壓力分類為「低壓力/中壓力/高壓力」三組
  - 缺失值處理與離群值移除
  - 性別 One-hot 編碼
- `scale_features(df, features)`: 特徵標準化
- `preprocess(path)`: 整合所有前處理步驟

### `src/db_utils.py`

MySQL 資料庫操作模組，支援自動建立資料庫，包含：

- `get_engine(mysql_uri)`: 建立 SQLAlchemy 引擎並自動處理連接錯誤
- `test_connection(mysql_uri)`: 測試資料庫連接是否正常
- `export_to_mysql(df, table_name, mysql_uri)`: 將 DataFrame 匯出至 MySQL
- `create_schema(sql_file_path, mysql_uri)`: 執行 SQL 檔案以建立資料庫結構
- `get_depression_by_pressure(mysql_uri)`: 查詢各壓力層級的憂鬱比例
- `get_pressure_by_degree(mysql_uri)`: 查詢各學歷層級的學業壓力平均值
- `get_pressure_stats_by_gender(mysql_uri)`: 查詢各性別的學業壓力統計

### `src/plot_utils.py`

視覺化函數模組，包含：

- `setup_chinese_font(font_path)`: 設定中文字型
- `plot_depression_by_pressure_level(df, zh_font)`: 繪製壓力層級憂鬱比例條形圖
- `plot_pressure_bins_distribution(df, zh_font, bins)`: 繪製連續壓力值分箱圖
- `plot_feature_importance_bar(importances, zh_font, title)`: 繪製特徵重要性條形圖
- `plot_confusion_matrix(cm, zh_font, title)`: 繪製混淆矩陣熱圖
- `plot_roc_curves(roc_data, zh_font)`: 繪製 ROC 曲線
- `plot_combined_depression_charts(df, zh_font)`: 組合多個圖表於單一畫面

### `src/model_utils.py`

模型訓練與評估模組，包含：

- `prepare_features(df, features, target)`: 準備特徵與標籤並標準化
- `train_logistic_regression(X_train, y_train)`: 訓練 Logistic Regression 模型
- `train_random_forest(X_train, y_train)`: 訓練 Random Forest 模型
- `evaluate_model(model, X_test, y_test)`: 評估模型效能（準確率、AUC、混淆矩陣）
- `get_feature_importance(model, X, y, features)`: 計算特徵重要性
- `cross_validate_models(X, y, features)`: 交叉驗證比較不同模型
- `check_correlation_with_depression(df)`: 檢查各特徵與憂鬱風險的相關性
- `train_and_evaluate(df, features)`: 整合式模型訓練與評估流程

### `src/font_loader.py`

字體下載與管理模組，支援跨平台：

- `get_font_directory()`: 獲取字體目錄路徑
- `download_noto_sans_cjk()`: 從網路下載 Noto Sans CJK 字體
- `find_system_noto_font()`: 在系統中尋找已安裝的字體
- `find_fallback_font()`: 尋找備用字體
- `download_font_if_not_exist()`: 檢查字體是否存在，若不存在則下載或使用備用字體

### `run_analysis.py`

主控程式，整合所有功能：

- `setup_environment()`: 設定執行環境與字型
- `parse_args()`: 解析命令列參數
- `run_basic_analysis(df, zh_font)`: 執行基本資料分析
- `export_to_db(df)`: 將資料匯出至 MySQL
- `run_model_analysis(df, zh_font)`: 執行模型訓練與評估
- `main()`: 主程式流程

---

## 核心分析結果

本研究發現：

1. **學業壓力與憂鬱風險具有中度正相關**（r = 0.475）
2. **不同壓力層級的憂鬱比例顯著不同**：
   - 低壓力組：19.44%
   - 中壓力組：52.03%
   - 高壓力組：81.63%
3. **卡方檢定顯示壓力層級間差異顯著**（χ² = 5740.656, df = 2, p < 0.001）
4. **模型預測效能**：
   - Logistic Regression：準確率 73.4%、AUC 0.805
   - Random Forest：準確率 69.0%、AUC 0.751
5. **兩種模型均將學業壓力視為最重要特徵**

---

## 套件依賴

```
pandas        # 資料處理與分析
numpy         # 數值計算
scipy         # 科學計算與統計分析
matplotlib    # 繪圖基礎庫
seaborn       # 統計資料視覺化
scikit-learn  # 機器學習模型
sqlalchemy    # ORM 資料庫工具
pymysql       # MySQL 連接驅動
requests      # HTTP 請求（用於字體下載）
```

---

## 貢獻指南

本專案歡迎各種形式的貢獻，包括但不限於：

1. 報告 Bug
2. 提交新功能與改進
3. 撰寫或改進文檔
4. 提供用例與範例

請在提交 Pull Request 前，先確保：

- 代碼風格遵循 PEP 8
- 所有測試通過
- 添加必要的文檔或註釋

---

## 授權協議

本專案採用 MIT 授權協議，詳情請見 LICENSE 文件。

---

## 報告撰寫指南

請在 `report/大學報告.md` 中使用下列章節結構：

1. 摘要
2. 緒論
3. 研究方法與資料
4. 探索性分析與視覺化
5. 推論性分析
6. 預測模型與比較
7. 結論與建議
8. 附錄（SQL 建表、Grafana Panel）

將程式產出的表格、圖表與數值對應到章節，以確保「程式 → 報告」完全一致。

---

如有任何問題或建議，歡迎開 Issue 或提出 Pull Request！
