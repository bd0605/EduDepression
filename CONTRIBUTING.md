# 貢獻指南

感謝您對 EduDepression 專案的興趣！我們歡迎各種形式的貢獻。

## 🌟 如何貢獻

### 報告問題

如果您發現了 bug 或有改進建議：

1. 先檢查 [Issues](https://github.com/bd0605/EduDepression/issues) 是否已有相似問題
2. 建立新的 Issue，詳細描述：
   - 問題的重現步驟
   - 預期行為 vs 實際行為
   - 您的環境資訊（作業系統、Python 版本等）
   - 錯誤訊息或截圖

### 提交程式碼

1. **Fork** 本專案到您的 GitHub 帳號
2. **Clone** 您的 Fork 到本地：
   ```bash
   git clone https://github.com/您的用戶名/EduDepression.git
   ```
3. 建立新的功能分支：
   ```bash
   git checkout -b feature/您的功能名稱
   ```
4. 進行您的修改
5. 確保程式碼品質：

   ```bash
   # 檢查程式碼風格
   flake8 .

   # 測試模組匯入
   python -c "from src import preprocess, plot_utils, model_utils, db_utils, font_loader"
   ```

6. 提交您的修改：
   ```bash
   git add .
   git commit -m "功能: 新增您的功能描述"
   ```
7. 推送到您的 Fork：
   ```bash
   git push origin feature/您的功能名稱
   ```
8. 建立 **Pull Request**

## 📋 程式碼規範

### Python 程式碼風格

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 規範
- 使用 4 個空格縮排
- 行長度限制為 127 字元
- 函數和類別需要適當的文檔字串

### 提交訊息格式

使用以下格式：

```
類型: 簡短描述

詳細描述（可選）

修正 #Issue編號（如適用）
```

類型可以是：

- `功能`: 新功能
- `修復`: Bug 修復
- `文檔`: 文檔更新
- `重構`: 程式碼重構
- `測試`: 測試相關

### 分支命名

- `feature/功能名稱`: 新功能
- `bugfix/問題描述`: Bug 修復
- `docs/文檔主題`: 文檔更新

## 🧪 測試

在提交 PR 前，請確保：

1. 所有現有功能仍正常運作
2. 新增功能有適當的測試
3. 程式碼通過 linting 檢查
4. 文檔已更新（如需要）

## 📝 文檔

如果您的貢獻涉及：

- 新功能：請更新 README.md 和相關文檔
- API 變更：請更新函數的 docstring
- 新的依賴：請更新 requirements.txt

## 🎯 開發環境設定

1. Fork 並 clone 專案
2. 建立虛擬環境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # 或
   venv\Scripts\activate     # Windows
   ```
3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   pip install flake8  # 用於程式碼檢查
   ```

## 🏷️ 版本發布

版本號遵循 [語義化版本](https://semver.org/lang/zh-TW/) 格式：

- `MAJOR.MINOR.PATCH`
- `MAJOR`: 不相容的 API 修改
- `MINOR`: 向下相容的功能新增
- `PATCH`: 向下相容的問題修正

## 💬 社群

- 在 Issues 中討論問題和想法
- 保持友善和建設性的交流
- 幫助其他貢獻者

## 📄 授權

貢獻的程式碼將採用與專案相同的 [MIT License](LICENSE) 授權。

---

再次感謝您的貢獻！ 🎉
