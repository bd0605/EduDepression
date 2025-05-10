-- 版本：v1.0.0
-- 建立資料庫
DROP DATABASE IF EXISTS student_depression;
CREATE DATABASE student_depression CHARACTER SET utf8mb4;
USE student_depression;

-- 建立主表（簡化版）
CREATE TABLE depression_data (
  id INT AUTO_INCREMENT PRIMARY KEY,
  gender VARCHAR(10),
  age FLOAT,
  degree VARCHAR(50),
  academic_pressure INT,
  work_pressure INT,
  cgpa FLOAT,
  study_satisfaction INT,
  depression TINYINT(1),
  INDEX idx_degree (degree),
  INDEX idx_depression (depression)
);

-- 建立學歷統計視圖
CREATE VIEW degree_depression_stats AS
SELECT degree,
       COUNT(*) AS total,
       SUM(depression) AS depressed,
       ROUND(100*AVG(depression),2) AS depression_pct
FROM depression_data
GROUP BY degree
ORDER BY depression_pct DESC;

-- Grafana使用者權限
GRANT SELECT ON student_depression.* TO 'grafana'@'localhost' IDENTIFIED BY 'grafana123';
FLUSH PRIVILEGES;
