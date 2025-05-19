# MySQL + Grafana 整合指南

本文件提供 EduDepression 專案與 MySQL 及 Grafana 整合的詳細步驟，以建立互動式視覺化儀表板，用於分析學業壓力與憂鬱風險的關聯性。

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

### 4. 檢查資料表結構與視圖

資料匯入後，您可以在 phpMyAdmin 中檢查以下資料表與視圖是否已正確建立：

- `student_depression` 表 - 主要資料表
- `v_depression_by_pressure` 視圖 - 壓力層級憂鬱比例
- `v_pressure_by_degree` 視圖 - 學歷層級壓力與憂鬱
- `v_stats_by_gender` 視圖 - 各性別壓力與憂鬱
- `v_pressure_depression_heatmap` 視圖 - 熱力圖資料
- `v_pressure_value_vs_depression` 視圖 - 壓力值與憂鬱率
- `v_cross_analysis` 視圖 - 交叉分析

## 二、設定 Python 環境與資料匯入

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

執行後會顯示匯出成功的訊息，此時資料已經導入 MySQL 資料庫中。

### 4. 資料匯入疑難排解

如果遇到資料匯入問題，可以檢查以下幾點：

1. MySQL 服務是否正在運行
2. 連接字串是否正確（預設為 `root@localhost:3306/depression_db`）
3. 使用 phpMyAdmin 手動檢查資料庫與資料表是否已正確建立

## 三、安裝與設定 Grafana

### 1. 安裝 Grafana

**Windows**:

1. 下載 Grafana：https://grafana.com/grafana/download?platform=windows
2. 執行安裝程式
3. 啟動 Grafana 服務（可透過 Windows 服務控制台或安裝目錄中的 `bin/grafana-server.exe`）

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

安裝完成後，Grafana 服務會在 http://localhost:3000 上運行。

### 2. 設定 MySQL 資料來源

1. 開啟瀏覽器，輸入 `http://localhost:3000/`
2. 使用預設帳號登入（帳號：admin，密碼：admin）
   - 首次登入會要求更改密碼，請設定一個安全的新密碼
3. 點選左側選單「Configuration」（齒輪圖示）→「Data Sources」
4. 點選「Add data source」
5. 選擇「MySQL」
6. 填寫以下資訊：
   - Name: MySQL
   - Host: localhost:3306
   - Database: depression_db
   - User: root
   - Password: (您的 MySQL 密碼，若未設定則留空)
   - Session timezone: +08:00 (根據您的時區調整)
   - Max open connections: 100
   - Max idle connections: 100
   - Max connection lifetime: 14400 (4 小時)
7. 點選「Save & Test」確認連線成功
   - 若顯示綠色的「Database Connection OK」，表示連接成功
   - 若連接失敗，請檢查 MySQL 服務是否運行及連接設定是否正確

### 3. 匯入儀表板

1. 點選左側選單「Create」（+圖示）→「Import」
2. 選擇「Upload .json file」
3. 上傳 `docs/grafana_dashboard.json` 檔案
4. 匯入設定：
   - Name: 學業壓力與憂鬱風險分析儀表板（可自訂）
   - Folder: General（可選擇其他資料夾）
   - Unique identifier (uid): edudepression（建議保留）
   - 資料來源下拉選單中選擇剛才建立的 MySQL 資料來源
5. 點選「Import」完成匯入

匯入成功後，將自動切換到儀表板檢視畫面。

## 四、使用 Grafana 儀表板

### 1. 儀表板面板說明

匯入完成後，儀表板將顯示以下視覺化內容：

1. **不同學業壓力水平的憂鬱比例**（左上）：

   - 條形圖顯示低、中、高三個壓力層級的憂鬱比例
   - Y 軸以百分比形式呈現（0-100%）
   - 使用 `v_depression_by_pressure` 視圖資料

2. **各學歷層級的學業壓力與憂鬱比例**（右上）：

   - 線圖同時顯示各學歷層級的平均壓力值（左 Y 軸）與憂鬱比例（右 Y 軸）
   - X 軸按照學歷由低到高排序：高中及以下、大學、碩士、博士
   - 使用 `v_pressure_by_degree` 視圖資料

3. **各性別學業壓力與憂鬱風險統計**（左下）：

   - 表格呈現各性別的樣本數、平均壓力值、壓力標準差與憂鬱比例
   - 憂鬱比例欄位按閾值著色：<40% 綠色，40-60% 黃色，>60% 紅色
   - 使用 `v_stats_by_gender` 視圖資料

4. **學業壓力 vs 憂鬱狀態熱力圖**（右下）：
   - 熱力圖展示不同壓力值與憂鬱狀態的分布情況
   - X 軸為壓力值（四捨五入），Y 軸為憂鬱狀態（0 或 1）
   - 顏色強度表示該組合的樣本數量
   - 直接查詢 `student_depression` 表

### 2. 互動功能使用

Grafana 儀表板提供多種互動功能：

1. **時間範圍選擇**：

   - 儀表板頂部的時間選擇器可調整視圖時間範圍
   - 本專案的資料沒有時間維度，因此此功能影響有限

2. **面板互動**：

   - 滑鼠懸停在圖表上可查看詳細數據
   - 點擊圖表某個元素可在其他面板中篩選相關資料（若有設定變數）
   - 面板右上角的選單可進行更多操作：全螢幕檢視、下載 CSV、編輯等

3. **儀表板設定**：
   - 點擊右上角齒輪圖示可設定儀表板屬性
   - 可調整自動重新整理間隔、時區等設定

## 五、自訂查詢與面板（進階）

您可以根據研究需求，建立其他視覺化面板：

### 1. 建立壓力分布直方圖

```sql
-- 查詢使用 v_pressure_value_vs_depression 視圖
SELECT
  pressure_value_bin,
  total_count
FROM
  v_pressure_value_vs_depression
ORDER BY
  pressure_value_bin;
```

### 2. 學歷與性別交叉分析

```sql
-- 查詢使用 v_cross_analysis 視圖
SELECT
  education,
  gender,
  avg_pressure,
  depression_rate,
  sample_count
FROM
  v_cross_analysis
ORDER BY
  education, gender;
```

### 3. 建立自訂面板步驟

1. 在儀表板中點選頂部的「Add panel」按鈕
2. 選擇面板類型（圖表、表格、熱力圖等）
3. 在查詢編輯器中選擇 MySQL 資料來源
4. 輸入 SQL 查詢或使用視圖
5. 設定視覺化選項（標題、軸標籤、顏色等）
6. 點選「Save」儲存面板

### 4. 設定儀表板變數（進階）

變數可以讓儀表板更具互動性，例如新增一個「學歷」變數：

1. 在儀表板設定中選擇「Variables」
2. 點選「New」建立新變數
3. 設定變數屬性：
   ```
   Name: education
   Type: Query
   Data source: MySQL
   Query: SELECT DISTINCT Degree4 FROM student_depression ORDER BY CASE Degree4 WHEN '高中及以下' THEN 1 WHEN '大學' THEN 2 WHEN '碩士' THEN 3 WHEN '博士' THEN 4 ELSE 5 END
   ```
4. 在查詢中使用變數：
   ```sql
   SELECT * FROM student_depression WHERE Degree4 = '$education'
   ```

## 六、常見問題與疑難排解

### 1. 無法連接到 MySQL

- 確認 MySQL 服務是否正在運行
- 檢查主機名稱、埠號、使用者名稱和密碼是否正確
- 確認 MySQL 使用者是否有權限存取資料庫

### 2. 圖表無資料顯示

- 使用 phpMyAdmin 確認資料表是否有資料
- 檢查 SQL 查詢是否正確
- 確認是否已正確配置視覺化設定

### 3. 時間相關查詢問題

- Grafana 的 MySQL 資料來源需要一個時間欄位，但本專案資料沒有時間維度
- 使用 `1 as time` 作為替代，並將面板時間範圍設為較寬的時間範圍（如「最近 6 小時」）

### 4. 性能優化

- 為經常查詢的欄位建立索引
- 使用視圖簡化複雜查詢
- 若資料量大，考慮設定 MySQL 查詢快取

## 七、資料安全與備份

### 1. 資料備份

定期備份 MySQL 資料庫是好習慣：

```bash
mysqldump -u root -p depression_db > depression_db_backup.sql
```

### 2. Grafana 儀表板備份

除了 JSON 檔案，也可從 Grafana 界面匯出儀表板：

1. 開啟儀表板
2. 點選「Share dashboard」（分享圖示）
3. 選擇「Export」標籤
4. 點選「Save to file」

## 結語

完成以上設定後，您將擁有一個互動式的視覺化儀表板，可以動態探索學業壓力與憂鬱風險的關聯性。此儀表板可作為校園心理健康監測與早期干預的重要工具。透過定期更新資料庫中的學生資料，可以進行長期趨勢分析，協助學校制定更有效的心理健康支持策略。

如有任何問題或需要進一步的定制化，請參考 Grafana 官方文檔：https://grafana.com/docs/
