
# 學生憂鬱症風險分析報告
## 版本：v1.0.0
## 日期：2025-05-11

## 1. 資料來源與內容
- **資料來源**：Student Depression Dataset
- **資料筆數**：____ 筆
- **資料欄位**：____ 個

### 主要欄位說明：
1. Gender（性別）
2. Age（年齡）
3. Degree（學歷）
4. Academic Pressure（學業壓力）
5. Work Pressure（工作壓力）
6. CGPA（學業成績）
7. Study Satisfaction（學習滿意度）
8. Depression（是否有憂鬱症）

## 2. 資料清洗過程
- 檢查缺失值：____個缺失值
- 處理異常值：____個異常值
- 資料標準化：使用StandardScaler進行特徵標準化

## 3. 分析結果

### 3.1 基礎統計分析
- 整體憂鬱症比例：____％
- 最高風險學歷：____（憂鬱率：____％）
- 最低風險學歷：____（憂鬱率：____％）

### 3.2 K-means聚類分析
- 群組數：3
- 群組0：樣本數=____，憂鬱比例=____％
- 群組1：樣本數=____，憂鬱比例=____％
- 群組2：樣本數=____，憂鬱比例=____％

### 3.3 邏輯回歸分析
- 模型準確率：____％
- AUC值：____
- 重要特徵：
  - Academic Pressure：係數=____
  - Work Pressure：係數=____
  - CGPA：係數=____
  - Study Satisfaction：係數=____

## 4. 結論與建議
（待填寫）

## 5. 資料視覺化（Grafana Dashboard）
- Depression Rate by Degree
- Total Students by Degree
- Overall Depression Rate

---
*報告生成時間：2025-05-11*
