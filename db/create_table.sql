-- MySQL 建表語法 for EduDepression 專案
-- 用於定義資料表結構與欄位型別

-- 如果 depression_db 資料庫不存在則建立
CREATE DATABASE IF NOT EXISTS depression_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用 depression_db 資料庫
USE depression_db;

-- 如果資料表已存在則刪除（可選）
DROP TABLE IF EXISTS student_depression;

-- 建立 student_depression 資料表
CREATE TABLE student_depression (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    -- 原始欄位
    Age FLOAT,
    Gender VARCHAR(50),
    Degree VARCHAR(100),
    Degree4 VARCHAR(50),
    degree_ord4 INT,
    Academic_Pressure FLOAT,
    Work_Pressure FLOAT,
    CGPA FLOAT,
    Study_Satisfaction FLOAT,
    
    -- 派生欄位
    Academic_Pressure_Value FLOAT,
    Academic_Pressure_Category VARCHAR(50),
    
    -- 目標變數
    Depression TINYINT(1),
    
    -- 其他資訊欄位（可選）
    Gender_Female TINYINT(1),
    Gender_Male TINYINT(1),
    Gender_Other TINYINT(1),
    
    -- 時間戳記（資料匯入時間）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- 索引
    INDEX idx_pressure_category (Academic_Pressure_Category),
    INDEX idx_degree (Degree4),
    INDEX idx_depression (Depression)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 建立視圖：壓力層級憂鬱比例
CREATE OR REPLACE VIEW v_depression_by_pressure AS
SELECT 
    Academic_Pressure_Category AS pressure_level,
    COUNT(*) AS total_count,
    SUM(Depression) AS depression_count,
    ROUND(SUM(Depression) / COUNT(*), 4) AS depression_rate
FROM 
    student_depression
GROUP BY 
    pressure_level
ORDER BY 
    CASE pressure_level
        WHEN '低壓力' THEN 1
        WHEN '中壓力' THEN 2
        WHEN '高壓力' THEN 3
        ELSE 4
    END;

-- 建立視圖：學歷層級壓力與憂鬱
CREATE OR REPLACE VIEW v_pressure_by_degree AS
SELECT 
    Degree4 AS degree_level,
    COUNT(*) AS total_count,
    ROUND(AVG(Academic_Pressure_Value), 2) AS avg_pressure,
    ROUND(SUM(Depression) / COUNT(*), 4) AS depression_rate
FROM 
    student_depression
GROUP BY 
    degree_level
ORDER BY 
    CASE degree_level
        WHEN '高中及以下' THEN 1
        WHEN '大學' THEN 2
        WHEN '碩士' THEN 3
        WHEN '博士' THEN 4
        ELSE 5
    END;

-- 建立視圖：各性別壓力與憂鬱
CREATE OR REPLACE VIEW v_stats_by_gender AS
SELECT 
    Gender AS gender,
    COUNT(*) AS total_count,
    ROUND(AVG(Academic_Pressure_Value), 2) AS avg_pressure,
    ROUND(STDDEV(Academic_Pressure_Value), 2) AS std_pressure,
    ROUND(SUM(Depression) / COUNT(*), 4) AS depression_rate
FROM 
    student_depression
GROUP BY 
    gender;

-- Grafana 查詢範例：壓力變數值與憂鬱率
-- CREATE OR REPLACE VIEW v_pressure_value_vs_depression AS
-- SELECT
--     FLOOR(Academic_Pressure_Value) AS pressure_value_bin,
--     COUNT(*) AS total_count,
--     ROUND(SUM(Depression) / COUNT(*), 4) AS depression_rate
-- FROM
--     student_depression
-- GROUP BY
--     pressure_value_bin
-- ORDER BY
--     pressure_value_bin;
