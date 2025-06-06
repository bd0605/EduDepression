# 📊 EduDepression Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

> 一個用於分析大學生學業壓力與憂鬱風險相關性的開源教學專案

本專案使用 27,901 筆大學生問卷資料，透過統計分析和機器學習方法，探討學業壓力與憂鬱風險之間的關聯性。專案採用模組化設計，適合用於教學、學習和研究用途。

## ✨ 主要特色

- 🔬 **完整統計分析流程**：從資料預處理到模型評估
- 📈 **豐富視覺化**：自動生成多種圖表和報告
- 🐍 **模組化架構**：易於理解和擴展的程式碼結構
- 🌐 **跨平台支援**：Windows、macOS、Linux 全支援
- 🗄️ **資料庫整合**：支援 MySQL 和 Grafana 視覺化
- 🎯 **教學友善**：適合大學生學習資料科學概念

## 📋 目錄

- [安裝指南](#安裝指南)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [核心功能](#核心功能)
- [使用教學](#使用教學)
- [貢獻指南](#貢獻指南)
- [授權資訊](#授權資訊)

## 🚀 安裝指南

### 系統需求

- Python 3.8 或更高版本
- 至少 4GB RAM
- 500MB 磁碟空間

### 克隆專案

```bash
git clone https://github.com/bd0605/EduDepression.git
cd EduDepression
```

### 建立虛擬環境

#### 🍎 macOS / Linux

```bash
# 使用 uv (推薦)
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用傳統方式
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 🪟 Windows

```cmd
REM 使用 uv (推薦)
pip install uv
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt

REM 或使用傳統方式
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ⚡ 快速開始

### 基本分析

```bash
python run_analysis.py
```

### 包含資料庫匯出

```bash
python run_analysis.py --to-mysql
```

第一次執行時，程式會自動：

- 下載並設定中文字體
- 進行資料預處理
- 執行統計分析
- 訓練機器學習模型
- 生成視覺化圖表

### 🧪 測試安裝

確認專案設定是否正確：

```bash
python test_project.py
```

這個測試腳本會檢查：

- 所有依賴套件是否安裝
- 模組是否能正常匯入
- 資料檔案是否存在
- 字體載入是否正常
- 基本功能是否運作

## 📁 專案結構

```
EduDepression/
├── 📊 data/
│   └── student_depression_dataset.csv  # Kaggle 資料集
├── 🗄️ db/
│   └── create_table.sql                # MySQL 建表語法
├── 📚 docs/
│   ├── mysql_grafana_guide.md          # MySQL與Grafana整合指南
│   ├── one-hot-gender-encoding.md      # 特徵工程說明
│   └── 重構記錄.md                     # 開發紀錄
├── 📈 grafana/
│   └── dashboard.json                  # Grafana 儀表板設定
├── 🏗️ src/
│   ├── __init__.py                     # 套件初始化
│   ├── preprocess.py                   # 資料前處理
│   ├── db_utils.py                     # 資料庫操作
│   ├── plot_utils.py                   # 視覺化工具
│   ├── model_utils.py                  # 機器學習模型
│   └── font_loader.py                  # 字體管理

├── colab_analysis.py                   # Google Colab 版本
├── index.html                          # 網頁分析報告
├── report.html                         # 詳細分析報告
├── run_analysis.py                     # 主程式
├── test_project.py                     # 專案完整性測試
├── requirements.txt                    # Python 依賴
├── .gitignore                          # Git 忽略檔案
├── .github/workflows/ci.yml            # GitHub Actions CI
├── CONTRIBUTING.md                     # 貢獻指南
├── LICENSE                             # MIT 授權
└── README.md                           # 本文件
```

## 🔧 核心功能

### 1. 資料預處理 (`src/preprocess.py`)

- ✅ 自動載入和清理資料
- ✅ 缺失值處理
- ✅ 特徵工程（學業壓力分級、性別編碼）
- ✅ 資料標準化

### 2. 統計分析

- 📊 描述性統計
- 🔗 相關性分析
- 📈 卡方檢定
- 🎯 主成分分析 (PCA)

### 3. 機器學習模型 (`src/model_utils.py`)

- 🤖 Logistic Regression
- 🌳 Random Forest
- 📊 交叉驗證
- 📈 ROC 曲線分析

### 4. 視覺化 (`src/plot_utils.py`)

- 📊 條形圖和直方圖
- 🔥 熱力圖
- 📈 ROC 曲線
- 🎨 混淆矩陣

### 5. 資料庫整合 (`src/db_utils.py`)

- 🗄️ MySQL 自動建表
- 📤 資料匯出
- 🔗 Grafana 整合

## 📖 使用教學

### 基本工作流程

1. **資料載入**: 程式自動載入 `data/student_depression_dataset.csv`
2. **前處理**: 清理資料、處理缺失值、特徵工程
3. **探索性分析**: 產生描述性統計和視覺化
4. **統計檢定**: 執行卡方檢定和相關性分析
5. **模型訓練**: 訓練並比較不同機器學習模型
6. **結果輸出**: 生成圖表和分析報告

### 學習範例

你可以直接使用主程式進行分析：

```bash
# 基本分析
python run_analysis.py

# 包含資料庫匯出的完整分析
python run_analysis.py --to-mysql
```

### 自訂分析

如果你想要執行特定的分析，可以直接使用各個模組：

```python
from src.preprocess import preprocess
from src.model_utils import train_and_evaluate
from src.plot_utils import plot_combined_depression_charts
from src.font_loader import download_font_if_not_exist

# 設定字體
font_path = download_font_if_not_exist()

# 載入和預處理資料
df = preprocess('data/student_depression_dataset.csv')

# 訓練模型
features = ['Academic Pressure_Value', 'CGPA', 'Work Pressure', 'degree_ord4']
results = train_and_evaluate(df, features)

# 生成視覺化
from matplotlib.font_manager import FontProperties
zh_font = FontProperties(fname=font_path)
plot_combined_depression_charts(df, zh_font)
```

### Google Colab 使用

如果你想在 Google Colab 中執行分析：

1. 上傳 `colab_analysis.py` 到 Colab
2. 上傳資料集到 `/content/student_depression_dataset.csv`
3. 執行整個檔案

### 網頁報告查看

- `index.html`: 互動式分析儀表板
- `report.html`: 詳細分析報告

### MySQL 和 Grafana 設定

1. **安裝 XAMPP**: [下載連結](https://www.apachefriends.org/)
2. **啟動 MySQL 服務**
3. **執行資料匯出**: `python run_analysis.py --to-mysql`
4. **設定 Grafana**: 參考 `docs/mysql_grafana_guide.md`

## 🎓 學習成果

完成本專案後，你將學會：

- 📊 資料科學專案的完整流程
- 🐍 Python 資料分析技能
- 📈 統計分析方法應用
- 🤖 機器學習模型建構
- 🗄️ 資料庫操作
- 📊 資料視覺化技巧

## 📚 資料來源

本專案使用的資料集來自 Kaggle，特此感謝：

> **Student Depression Dataset**  
> 作者: [@erimsaholut](https://www.kaggle.com/code/erimsaholut/student-depression-dataset/notebook)  
> 來源: [Kaggle Dataset](https://www.kaggle.com/code/erimsaholut/student-depression-dataset/notebook)

資料集包含 27,901 筆大學生問卷回應，涵蓋學業壓力、財務壓力、工作壓力等多個面向的資訊。

## 🤝 貢獻指南

我們歡迎所有形式的貢獻！

### 如何貢獻

1. **Fork** 本專案
2. 建立你的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的改動 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 建立 **Pull Request**

### 開發指南

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 程式碼風格
- 為新功能添加適當的測試
- 更新相關文檔
- 確保所有測試通過

### 問題回報

如果你發現了 bug 或有功能建議，請：

1. 檢查是否已有相關的 Issue
2. 建立新的 Issue 並詳細描述問題
3. 提供重現步驟（如果是 bug）

## 🆘 常見問題

### Q: 程式執行時出現字體錯誤？

A: 程式會自動下載中文字體，確保網路連線正常即可。

### Q: 如何修改分析的特徵？

A: 編輯 `run_analysis.py` 中的 `features` 列表。

### Q: 可以使用自己的資料集嗎？

A: 可以，只需確保資料格式與範例資料集一致。

## 📄 授權資訊

本專案採用 [MIT License](LICENSE) 授權。

## 🌟 Star History

如果這個專案對你有幫助，請給我們一個 ⭐！

---

**🎯 教學目標**: 幫助大學生理解資料科學的完整流程，從資料清理到機器學習模型建構。

**🔧 技術棧**: Python, pandas, scikit-learn, matplotlib, seaborn, MySQL, Grafana

**👨‍💻 維護者**: [@bd0605](https://github.com/bd0605)
