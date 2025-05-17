## 🐍 One-Hot Encoding（Dummy 變數）是什麼？

機器學習模型無法直接處理文字資料（如 `Gender = 'Male' / 'Female' / 'Other'`），
所以我們會將這些**類別資料轉換為數值欄位**，每個類別會變成一個 dummy 欄位，稱作 One-Hot Encoding。

**原始資料範例：**

```csv
Gender
Male
Female
Other
```

**轉換後（完整 One-Hot）：**

```csv
Gender_Male | Gender_Female | Gender_Other
1           | 0             | 0
0           | 1             | 0
0           | 0             | 1
```

---

## ⚠️ 為什麼不保留全部欄位？

若保留全部類別的 dummy 欄位，會導致**多重共線性（Multicollinearity）**，
特別是在 Logistic Regression 中，這會讓模型不穩定或參數難以解釋。

因此通常會設定 `drop_first=True`，丟棄其中一個類別（例如 `Male`），作為**基準類別**。

---

## ✅ 實際範例：保留 Gender_Female 與 Gender_Other

**保留欄位：**

- `Gender_Female`
- `Gender_Other`

**結果含義對照：**

| Gender_Female | Gender_Other | 實際性別         |
| ------------- | ------------ | ---------------- |
| 0             | 0            | Male（基準類別） |
| 1             | 0            | Female           |
| 0             | 1            | Other            |

---

## 🧪 對應的程式碼片段

```python
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
```

這行程式碼會將 `Gender` 欄位轉為兩個欄位：`Gender_Female`, `Gender_Other`，
而將 `Male` 當作基準類別省略。

---

## 📊 應用實例：學業壓力預測憂鬱風險

在你的分析中，這些 dummy 欄位會與其他特徵一起放進模型，例如：

```python
features = ['Academic Pressure_Value', 'degree_ord4'] +
           [c for c in df.columns if c.startswith('Gender_')]
```

這樣模型就能分辨：
某人是 Female 還是 Other，相對於基準 Male 的風險差異。
