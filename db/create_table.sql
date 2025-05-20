-- MySQL 建表語法 for EduDepression 專案
-- 用於定義資料表結構與欄位型別，支援 Grafana 儀表板視覺化

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
    INDEX idx_depression (Depression),
    INDEX idx_pressure_value (Academic_Pressure_Value)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

