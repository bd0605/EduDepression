# XAMPP + Grafana 設定指南

版本：v1.0.0  
更新日期：2025-05-11

## XAMPP 設定

### 1. 安裝XAMPP
- 下載連結：https://www.apachefriends.org/
- 選擇適合您作業系統的版本
- 安裝時勾選 Apache 和 MySQL

### 2. 啟動服務
1. 開啟 XAMPP Control Panel
2. 啟動 Apache（非必要）
3. 啟動 MySQL（必要）

### 3. 建立資料庫
1. 開啟 phpMyAdmin：http://localhost/phpmyadmin
2. 建立資料庫：
   ```sql
   CREATE DATABASE student_depression CHARACTER SET utf8mb4;
   ```
3. 執行初始化：
   ```sql
   source /path/to/db/init_student_depression.sql
   ```

## Grafana 設定

### 1. 安裝 Grafana
- Windows: 下載安裝檔
- Mac: `brew install grafana`
- Linux: 參考官方文檔

### 2. 啟動 Grafana
```bash
grafana-server
```
預設網址：http://localhost:3000  
預設帳密：admin/admin

### 3. 新增 MySQL 資料源

1. 進入 Configuration → Data Sources
2. 點選 Add data source → MySQL
3. 設定參數：
   - Host: localhost:3306
   - Database: student_depression
   - User: grafana
   - Password: grafana123

### 4. 建立 Dashboard

1. 建立新 Dashboard
2. 新增 Panel，選擇查詢：
   ```sql
   SELECT degree, depression_pct 
   FROM degree_depression_stats
   ```
3. 選擇適合的視覺化類型（Bar Chart、Pie Chart等）

## 常用查詢範例

### 學歷憂鬱率
```sql
SELECT degree, depression_pct 
FROM degree_depression_stats
ORDER BY depression_pct DESC
```

### 整體統計
```sql
SELECT 
  COUNT(*) as total_students,
  SUM(depression) as depressed_count,
  ROUND(AVG(depression)*100,1) as depression_rate
FROM depression_data
```

### 性別統計
```sql
SELECT 
  gender,
  COUNT(*) as count,
  ROUND(AVG(depression)*100,1) as depression_rate
FROM depression_data
GROUP BY gender
```

## 故障排除

1. **MySQL連接錯誤**
   - 確認MySQL服務已啟動
   - 檢查使用者權限

2. **Grafana連接失敗**
   - 確認資料庫名稱正確
   - 確認使用者帳密

3. **無資料顯示**
   - 確認已執行 `run_analysis.py`
   - 檢查SQL查詢語法