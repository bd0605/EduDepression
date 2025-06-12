# XAMPP/MySQL/Grafana 完整環境設定指南

本指南將協助您從零開始建立完整的資料分析環境，包含 XAMPP、MySQL 資料庫及 Grafana 視覺化平台的安裝與設定，用於 EduDepression 專案的學業壓力與憂鬱風險分析。

## 📋 目錄

1. [前置需求](#前置需求)
2. [XAMPP 與 MySQL 設定](#xampp-與-mysql-設定)
3. [Python 環境配置](#python-環境配置)
4. [Grafana 安裝與設定](#grafana-安裝與設定)
5. [資料視覺化設定](#資料視覺化設定)
6. [疑難排解](#疑難排解)
7. [進階功能](#進階功能)

## 前置需求

在開始之前，請確保您的系統已安裝以下軟體：

- **作業系統**: Windows 10/11, macOS 10.15+, 或 Ubuntu 18.04+
- **Python**: 3.8 或更新版本
- **磁碟空間**: 至少 2GB 可用空間
- **記憶體**: 建議 4GB 以上 RAM

## XAMPP 與 MySQL 設定

### 步驟 1: 下載並安裝 XAMPP

1. 前往 [XAMPP 官方網站](https://www.apachefriends.org/) 下載適合您作業系統的版本
2. 執行安裝程式，確保選擇以下元件：
   - ✅ **Apache** (網頁伺服器)
   - ✅ **MySQL** (資料庫伺服器)
   - ✅ **phpMyAdmin** (資料庫管理工具)

### 步驟 2: 啟動 MySQL 服務

1. 開啟 **XAMPP Control Panel**
2. 點擊 MySQL 旁的 **"Start"** 按鈕
3. 確認狀態顯示為 "Running" (綠色背景)

> 💡 **小提示**: 如果 MySQL 無法啟動，可能是連接埠 3306 被其他程式占用。您可以在 XAMPP 中點擊 "Config" → "my.ini" 來修改連接埠設定。

### 步驟 3: 建立資料庫

**方法一：使用 phpMyAdmin（推薦給初學者）**

1. 開啟瀏覽器，前往 `http://localhost/phpmyadmin/`
2. 點擊左側的 **"新增"** 建立資料庫
3. 資料庫名稱輸入：`depression_db`
4. 排序規則選擇：`utf8mb4_unicode_ci`
5. 點擊 **"建立"**

**方法二：使用命令列**

```bash
# 連接到 MySQL
mysql -u root -p

# 建立資料庫
CREATE DATABASE depression_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 步驟 4: 驗證資料庫設定

在 phpMyAdmin 中應該能看到新建立的 `depression_db` 資料庫。點擊該資料庫，確認可以正常訪問。

## Python 環境配置

### 步驟 1: 建立虛擬環境

在專案根目錄下執行以下指令：

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 步驟 2: 安裝相依套件

```bash
# 安裝所有必要套件
pip install -r requirements.txt

# 驗證安裝
python test_project.py
```

### 步驟 3: 匯出資料至 MySQL

執行以下指令將分析結果匯出至 MySQL：

```bash
python run_analysis.py --to-mysql
```

執行過程中，您將看到詳細的進度資訊：

```
🚀 EduDepression 學業壓力與憂鬱風險分析系統
============================================================

🔧 環境設定: 設定字型與執行環境
  → 下載中文字型檔案

📊 資料預處理: 載入與清理資料
  → 讀取CSV檔案
  處理後資料集大小: (27873, 15)

📈 統計分析: 計算相關性與統計檢定
  學業壓力與憂鬱風險的相關係數: 0.475

🤖 模型訓練: 機器學習模型訓練與評估
  → 訓練Logistic Regression和Random Forest模型

🗄️ 資料匯出: 匯出資料至MySQL資料庫
  → 測試MySQL資料庫連接
  ✅ 資料成功匯出至 MySQL！
```

成功匯出後，可以在 phpMyAdmin 中查看 `student_depression` 資料表及相關視圖。

## Grafana 安裝與設定

### 步驟 1: 安裝 Grafana

**Windows 系統:**

1. 前往 [Grafana 下載頁面](https://grafana.com/grafana/download?platform=windows)
2. 下載 Windows 安裝程式
3. 執行安裝程式，使用預設設定即可

**macOS 系統:**

```bash
# 使用 Homebrew 安裝
brew update
brew install grafana

# 啟動 Grafana 服務
brew services start grafana
```

**Linux (Ubuntu/Debian) 系統:**

```bash
# 新增 Grafana 官方套件庫
sudo apt-get install -y software-properties-common
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"

# 安裝 Grafana
sudo apt-get update
sudo apt-get install grafana

# 啟動服務
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

### 步驟 2: 首次登入設定

1. 開啟瀏覽器，前往 `http://localhost:3000/`
2. 使用預設帳號登入：
   - 帳號：`admin`
   - 密碼：`admin`
3. 系統會要求設定新密碼，請選擇一個安全的密碼

### 步驟 3: 設定 MySQL 資料來源

1. 點擊左側齒輪圖示 ⚙️ → **"Data Sources"**
2. 點擊 **"Add data source"**
3. 選擇 **"MySQL"**
4. 填寫連接資訊：

```
Name: EduDepression-MySQL
Host: localhost:3306
Database: depression_db
Username: root
Password: (您的MySQL密碼，通常為空)
Session timezone: +08:00
```

5. 點擊 **"Save & Test"** 確認連接成功

> ✅ **成功訊息**: 如果看到綠色的 "Database Connection OK"，表示設定正確。

## 資料視覺化設定

### 步驟 1: 匯入預設儀表板

1. 點擊左側 **"Dashboards"** → 右側 **"New"** 展開後點擊 **"Import"**
2. 點擊 **"Upload .json file"**
3. 選擇專案中的 `grafana/dashboard.json` 檔案
4. 設定儀表板資訊：
   - **Name**: 學業壓力與憂鬱風險分析儀表板
   - **Folder**: General
   - **Data source**: 選擇剛才建立的 EduDepression-MySQL
5. 點擊 **"Import"**

### 步驟 2: 瞭解儀表板面板

匯入成功後，您將看到以下視覺化面板：

#### 📊 不同學業壓力水平的憂鬱比例

- **資料來源**: `v_depression_by_pressure` 視圖
- **圖表類型**: 長條圖
- **顯示內容**: 低壓力、中壓力、高壓力三組的憂鬱比例
- **關鍵洞察**: 可以清楚看出壓力與憂鬱風險的正相關關係

#### 📈 各學歷層級的學業壓力與憂鬱比例

- **資料來源**: `v_pressure_by_degree` 視圖
- **圖表類型**: 組合圖（長條圖 + 折線圖）
- **顯示內容**: 不同學歷層級的平均壓力值與憂鬱比例
- **關鍵洞察**: 分析學歷與心理健康的關聯性

### 步驟 3: 自訂面板和查詢

如果您想新增自訂的視覺化面板：

1. 點擊儀表板右上角的 **"Add panel"**
2. 選擇面板類型（例如：Time series、Stat、Table 等）
3. 在查詢編輯器中輸入 SQL：

```sql
-- 範例：性別與憂鬱風險分析
SELECT
    Gender,
    COUNT(*) as total_count,
    SUM(Depression) as depression_count,
    ROUND(AVG(Depression) * 100, 1) as depression_rate
FROM student_depression
GROUP BY Gender
ORDER BY depression_rate DESC;
```

4. 設定視覺化選項（軸標籤、顏色、圖例等）
5. 點擊 **"Apply"** 儲存面板

## 疑難排解

### 常見問題 1: 無法連接到 MySQL

**症狀**: Grafana 顯示 "database connection failed"

**解決方案**:

1. 確認 XAMPP 中的 MySQL 服務正在運行
2. 檢查連接埠是否正確（預設為 3306）
3. 確認資料庫名稱拼寫正確
4. 檢查 MySQL 使用者權限

### 常見問題 2: 圖表沒有顯示資料

**症狀**: 面板顯示 "No data" 或空白

**解決方案**:

1. 確認已執行 `python run_analysis.py --to-mysql`
2. 在 phpMyAdmin 中檢查資料表是否有資料
3. 檢查 SQL 查詢語法是否正確
4. 確認時間範圍設定合適

### 常見問題 3: 中文字型顯示異常

**症狀**: 圖表中的中文顯示為方框或亂碼

**解決方案**:

1. 在 Grafana 中調整面板的字型設定
2. 確認瀏覽器支援中文字型
3. 可以修改查詢，使用英文標籤代替中文

### 常見問題 4: 效能問題

**症狀**: 儀表板載入緩慢

**解決方案**:

1. 為常用查詢欄位建立資料庫索引
2. 調整 Grafana 的重新整理頻率
3. 使用視圖簡化複雜查詢
4. 限制查詢的資料範圍

## 進階功能

### 建立警報規則

Grafana 支援設定警報，當特定指標超過閾值時發送通知：

1. 編輯面板，切換到 **"Alert"** 標籤
2. 點擊 **"Create Alert"**
3. 設定警報條件（例如：憂鬱比例超過 80%）
4. 配置通知管道（Email、Slack 等）

### 匯出和分享

**匯出儀表板**:

- 點擊 Share → Export → Save to file

**分享儀表板**:

- 點擊 Share → Link → 複製連結給其他使用者

**嵌入到網頁**:

- 點擊 Share → Embed → 複製 iframe 程式碼

### 自動備份

定期備份您的 Grafana 設定和 MySQL 資料：

```bash
# 備份 MySQL 資料
mysqldump -u root -p depression_db > backup_$(date +%Y%m%d).sql

# 備份 Grafana 設定（Linux/macOS）
sudo cp -r /etc/grafana/ ~/grafana_backup_$(date +%Y%m%d)/
```

## 🎯 學習成果驗證

完成本指南後，您應該能夠：

- ✅ 成功設定 XAMPP 和 MySQL 環境
- ✅ 將 Python 分析結果匯出至資料庫
- ✅ 安裝並設定 Grafana 視覺化平台
- ✅ 建立專業的資料分析儀表板
- ✅ 解釋學業壓力與憂鬱風險的視覺化結果
- ✅ 自訂查詢和建立新的視覺化面板

## 📚 延伸學習資源

- [Grafana 官方文檔](https://grafana.com/docs/)
- [MySQL 學習指南](https://dev.mysql.com/doc/)
- [資料視覺化最佳實務](https://www.tableau.com/learn/articles/data-visualization)
- [校園心理健康研究方法](https://www.apa.org/science/resources)

## 🤝 技術支援

如果在設定過程中遇到問題：

1. 查看專案的 [Issue 頁面](https://github.com/bd0605/EduDepression/issues)
2. 參考本指南的疑難排解章節
3. 在社群論壇尋求協助
4. 聯繫專案維護者
