# MySQL + Grafana 整合指南

本文件提供 EduDepression 專案與 MySQL 及 Grafana 整合的詳細步驟，以建立互動式視覺化儀表板。

## 一、設定 XAMPP 與 MySQL

### 1. 安裝 XAMPP

1. 下載 XAMPP：https://www.apachefriends.org/
2. 安裝時勾選以下元件：
   - Apache（必選）
   - MySQL（必選）
   - phpMyAdmin（建議）

### 2. 啟動 MySQL 服務

1. 開啟 XAMPP Control Panel
2. 啟動 MySQL 服務（按下 "Start" 按鈕）
3. 啟動 Apache 服務（若需使用 phpMyAdmin）

### 3. 建立資料庫與資料表

**方法一：使用 phpMyAdmin**

1. 開啟瀏覽器，輸入 `http://localhost/phpmyadmin/`
2. 點選「新增」建立名為 `depression_db` 的資料庫
   - 選擇排序規則：`utf8mb4_unicode_ci`
3. 選擇 `depression_db` 資料庫，切換到「SQL」標籤
4. 複製 `db/create_table.sql` 的內容並執行

**方法二：使用命令列**

1. 開啟命令提示字元/終端機
2. 使用以下指令連接 MySQL：
   ```bash
   mysql -u root -p
   ```
3. 執行以下指令：
   ```sql
   CREATE DATABASE depression_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   USE depression_db;
   SOURCE /path/to/EduDepression/db/create_table.sql;
   ```

## 二、設定 Python 環境

### 1. 建立虛擬環境

```bash
# 在專案根目錄下執行
python -m venv venv

# 啟動虛擬環境（Windows）
venv\Scripts\activate

# 啟動虛擬環境（macOS/Linux）
source venv/bin/activate
```

### 2. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 3. 資料匯入 MySQL

```bash
# 執行主程式並匯出資料至 MySQL
python run_analysis.py --to-mysql
```

## 三、安裝與設定 Grafana

### 1. 安裝 Grafana

**Windows**:

1. 下載 Grafana：https://grafana.com/grafana/download?platform=windows
2. 執行安裝程式
3. 啟動 Grafana 服務

**macOS**:

```bash
brew update
brew install grafana
brew services start grafana
```

**Linux (Ubuntu/Debian)**:

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana
sudo systemctl start grafana-server
```

### 2. 設定 MySQL 資料來源

1. 開啟瀏覽器，輸入 `http://localhost:3000/`
2. 使用預設帳號登入（帳號：admin，密碼：admin）
3. 點選左側選單「Configuration」（齒輪圖示）→「Data Sources」
4. 點選「Add data source」
5. 選擇「MySQL」
6. 填寫以下資訊：
   - Name: MySQL
   - Host: localhost:3306
   - Database: depression_db
   - User: root
   - Password: (您的 MySQL 密碼)
   - Session timezone: +08:00 (根據您的時區調整)
7. 點選「Save & Test」確認連線成功

### 3. 匯入儀表板

1. 點選左側選單「Create」（+圖示）→「Import」
2. 選擇「Upload .json file」
3. 上傳 `docs/grafana_dashboard.json` 檔案
4. 點選「Import」完成匯入

## 四、使用 Grafana 儀表板

匯入完成後，儀表板將自動顯示以下視覺化內容：

1. **不同學業壓力水平的憂鬱比例**：條形圖顯示低、中、高三個壓力層級的憂鬱比例
2. **各學歷層級的學業壓力與憂鬱比例**：線圖同時顯示各學歷的平均壓力值與憂鬱比例
3. **各性別學業壓力與憂鬱風險統計**：表格呈現各性別的樣本數、平均壓力值、壓力標準差與憂鬱比例
4. **學業壓力 vs 憂鬱狀態熱力圖**：熱力圖展示不同壓力值與憂鬱狀態的分布情況

## 五、自訂查詢與面板（進階）

您可以根據研究需求，建立其他視覺化面板：

### 1. 建立壓力分布直方圖

```sql
SELECT
  ROUND(Academic_Pressure_Value) AS pressure_bin,
  COUNT(*) AS count
FROM
  student_depression
GROUP BY
  pressure_bin
ORDER BY
  pressure_bin;
```

### 2. 多變數交叉分析

```sql
SELECT
  Degree4 AS education,
  Gender AS gender,
  ROUND(AVG(Academic_Pressure_Value), 2) AS avg_pressure,
  ROUND(SUM(Depression) / COUNT(*), 4) AS depression_rate,
  COUNT(*) AS sample_count
FROM
  student_depression
GROUP BY
  education, gender
ORDER BY
  education, gender;
```

### 3. 隨時間變化趨勢（若有時間欄位）

```sql
-- 假設有 record_date 欄位
SELECT
  DATE_FORMAT(record_date, '%Y-%m') AS month,
  ROUND(AVG(Academic_Pressure_Value), 2) AS avg_pressure,
  ROUND(SUM(Depression) / COUNT(*), 4) AS depression_rate
FROM
  student_depression
GROUP BY
  month
ORDER BY
  month;
```

## 結語

完成以上設定後，您將擁有一個互動式的視覺化儀表板，可以動態探索學業壓力與憂鬱風險的關聯性。此儀表板可作為校園心理健康監測與早期干預的重要工具。
