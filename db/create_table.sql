/* === 0. 建庫 & 切庫 ==================================================== */
CREATE DATABASE IF NOT EXISTS depression_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE depression_db;

/* === 1. 建表 ========================================================== */
DROP TABLE IF EXISTS student_depression;
CREATE TABLE student_depression (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,

  /* 原始欄位 ---------------------------------------------------------- */
  Age                 FLOAT,
  Gender              VARCHAR(50),
  Degree              VARCHAR(100),
  Degree4             VARCHAR(50),
  degree_ord4         INT,
  `Academic Pressure` FLOAT,
  `Work Pressure`     FLOAT,
  CGPA                FLOAT,
  `Study Satisfaction` FLOAT,

  /* 派生欄位（含空格，一律用反引號） ----------------------------------- */
  `Academic Pressure_Value`      FLOAT,
  `Academic Pressure_Category`   VARCHAR(50),

  /* 目標變數 ---------------------------------------------------------- */
  Depression          TINYINT(1),

  /* 其他資訊欄位 ------------------------------------------------------ */
  Gender_Female       TINYINT(1),
  Gender_Male         TINYINT(1),
  Gender_Other        TINYINT(1),

  /* 時間戳 ------------------------------------------------------------ */
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             ON UPDATE CURRENT_TIMESTAMP,

  /* 索引（也記得用反引號） ------------------------------------------- */
  INDEX idx_pressure_category (`Academic Pressure_Category`),
  INDEX idx_degree            (Degree4),
  INDEX idx_depression        (Depression),
  INDEX idx_pressure_value    (`Academic Pressure_Value`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci;

/* === 2. 視圖：壓力層級 × 憂鬱比例 (Panel 1) ========================= */
CREATE OR REPLACE VIEW v_depression_by_pressure AS
SELECT
  `Academic Pressure_Category`                   AS pressure_level,
  COUNT(*)                                       AS total_count,
  SUM(Depression)                                AS depression_count,
  ROUND(SUM(Depression) / COUNT(*), 4)           AS depression_rate
FROM student_depression
GROUP BY pressure_level
ORDER BY CASE pressure_level
  WHEN '低壓力' THEN 1
  WHEN '中壓力' THEN 2
  WHEN '高壓力' THEN 3
  ELSE 4
END;

/* === 3. 視圖：學歷層級壓力與憂鬱 (Panel 2) ========================= */
CREATE OR REPLACE VIEW v_pressure_by_degree AS
SELECT
  Degree4                                         AS degree_level,
  COUNT(*)                                        AS total_count,
  ROUND(AVG(`Academic Pressure_Value`), 2)        AS avg_pressure,
  ROUND(SUM(Depression) / COUNT(*), 4)            AS depression_rate
FROM student_depression
GROUP BY degree_level
ORDER BY CASE degree_level
  WHEN '高中及以下' THEN 1
  WHEN '大學'       THEN 2
  WHEN '碩士'       THEN 3
  WHEN '博士'       THEN 4
  ELSE 5
END;

/* === 4. 視圖：各性別壓力與憂鬱 (Panel 3) =========================== */
CREATE OR REPLACE VIEW v_stats_by_gender AS
SELECT
  Gender                                           AS gender,
  COUNT(*)                                         AS total_count,
  ROUND(AVG(`Academic Pressure_Value`), 2)         AS avg_pressure,
  ROUND(STDDEV(`Academic Pressure_Value`), 2)      AS std_pressure,
  ROUND(SUM(Depression) / COUNT(*), 4)             AS depression_rate
FROM student_depression
GROUP BY gender;

/* === 5. 視圖：壓力值 × 憂鬱狀態熱力圖 (Panel 4) ==================== */
CREATE OR REPLACE VIEW v_pressure_depression_heatmap AS
SELECT
  ROUND(`Academic Pressure_Value`)                 AS pressure_value_bin,
  Depression                                       AS depression,
  COUNT(*)                                         AS value
FROM student_depression
GROUP BY pressure_value_bin, depression
ORDER BY pressure_value_bin, depression;

/* === 6. 視圖：壓力值 × 憂鬱率 (延伸自訂) ============================ */
CREATE OR REPLACE VIEW v_pressure_value_vs_depression AS
SELECT
  FLOOR(`Academic Pressure_Value`)                 AS pressure_value_bin,
  COUNT(*)                                         AS total_count,
  ROUND(SUM(Depression) / COUNT(*), 4)             AS depression_rate
FROM student_depression
GROUP BY pressure_value_bin
ORDER BY pressure_value_bin;

/* === 7. 視圖：學歷 × 性別交叉分析 (延伸自訂) ======================= */
CREATE OR REPLACE VIEW v_cross_analysis AS
SELECT
  Degree4                                          AS education,
  Gender                                           AS gender,
  ROUND(AVG(`Academic Pressure_Value`), 2)         AS avg_pressure,
  ROUND(STDDEV(`Academic Pressure_Value`), 2)      AS std_pressure,
  ROUND(SUM(Depression) / COUNT(*), 4)             AS depression_rate,
  COUNT(*)                                         AS sample_count
FROM student_depression
GROUP BY education, gender
ORDER BY CASE education
  WHEN '高中及以下' THEN 1
  WHEN '大學'       THEN 2
  WHEN '碩士'       THEN 3
  WHEN '博士'       THEN 4
  ELSE 5
END, gender;
