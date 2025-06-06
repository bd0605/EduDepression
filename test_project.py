#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EduDepression 專案完整性測試

這個腳本用於驗證專案的各個模組是否能正常匯入和執行基本功能。
適合在新環境中快速檢查專案是否設定正確。
"""

import sys
import os
import traceback
from tqdm import tqdm

class TestProgressTracker:
    """統一的測試進度追蹤器"""
    
    def __init__(self):
        self.total_tests = 5
        self.current_test = 0
        self.pbar = None
        self.results = []
        
        # 測試項目定義
        self.tests = [
            ("📦 依賴套件", "檢查必要的Python套件"),
            ("🔧 模組匯入", "驗證自定義模組載入"),
            ("📂 資料檔案", "確認資料集檔案存在"),
            ("🔤 字體載入", "測試中文字型下載與載入"),
            ("⚙️ 基本功能", "驗證核心功能運作")
        ]
    
    def start(self):
        """開始測試進度追蹤"""
        print("🚀 EduDepression 專案完整性測試")
        print("=" * 50)
        self.pbar = tqdm(
            total=100, 
            desc="準備測試", 
            unit="%", 
            ncols=80,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}'
        )
    
    def next_test(self, detail=""):
        """進入下一個測試"""
        if self.pbar is None:
            return
            
        self.current_test += 1
        test_name, test_desc = self.tests[self.current_test - 1]
        
        # 計算進度百分比
        progress = int((self.current_test - 1) / self.total_tests * 100)
        self.pbar.n = progress
        
        # 更新描述
        desc = f"{test_name}"
        if detail:
            desc += f" - {detail}"
        self.pbar.set_description(desc)
        self.pbar.refresh()
        
        # 在控制台顯示測試資訊
        print(f"\n{test_name}: {test_desc}")
    
    def update_detail(self, detail):
        """更新當前測試的詳細資訊"""
        if self.pbar is None:
            return
            
        test_name, _ = self.tests[self.current_test - 1]
        desc = f"{test_name} - {detail}"
        self.pbar.set_description(desc)
        self.pbar.refresh()
    
    def finish_test(self, result, test_name=""):
        """完成當前測試"""
        if self.pbar is None:
            return
            
        # 記錄結果
        if not test_name:
            test_name = self.tests[self.current_test - 1][0]
        self.results.append((test_name, result))
        
        # 更新到下一個測試的起始點
        progress = int(self.current_test / self.total_tests * 100)
        self.pbar.n = progress
        self.pbar.refresh()
    
    def complete(self):
        """完成所有測試"""
        if self.pbar is None:
            return
            
        self.pbar.n = 100
        self.pbar.set_description("✅ 測試完成")
        self.pbar.refresh()
        self.pbar.close()
        
        # 顯示測試結果總結
        self._show_summary()
    
    def _show_summary(self):
        """顯示測試結果總結"""
        print("\n" + "=" * 50)
        print("📊 測試結果總結:")
        
        passed = 0
        for test_name, result in self.results:
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"  {test_name.split(' ', 1)[1] if ' ' in test_name else test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 總計: {passed}/{len(self.results)} 項測試通過")
        
        if passed == len(self.results):
            print("🎉 所有測試通過！專案設定正確。")
        else:
            print("⚠️ 部分測試失敗，請檢查上述錯誤訊息。")

def test_dependencies(progress_tracker):
    """檢查依賴套件"""
    progress_tracker.next_test()
    
    required_packages = [
        "pandas", "numpy", "scipy", "matplotlib", 
        "seaborn", "sklearn", "sqlalchemy", "pymysql", "requests", "tqdm"
    ]
    
    missing_packages = []
    
    for i, package in enumerate(required_packages):
        progress_tracker.update_detail(f"檢查 {package}")
        try:
            if package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (未安裝)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  💡 請安裝缺少的套件: pip install {' '.join(missing_packages)}")
        result = False
    else:
        print(f"  ✅ 所有依賴套件已正確安裝")
        result = True
    
    progress_tracker.finish_test(result)
    return result

def test_imports(progress_tracker):
    """測試所有模組匯入"""
    progress_tracker.next_test()
    
    modules_to_test = [
        "src.preprocess",
        "src.plot_utils", 
        "src.model_utils",
        "src.db_utils",
        "src.font_loader"
    ]
    
    failed_imports = []
    
    for i, module in enumerate(modules_to_test):
        progress_tracker.update_detail(f"匯入 {module}")
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"  ⚠️ {module}: {e}")
            failed_imports.append(module)
    
    result = len(failed_imports) == 0
    if result:
        print(f"  ✅ 所有模組匯入成功")
    else:
        print(f"  ❌ {len(failed_imports)} 個模組匯入失敗")
    
    progress_tracker.finish_test(result)
    return result

def test_data_file(progress_tracker):
    """檢查資料檔案是否存在"""
    progress_tracker.next_test()
    
    data_path = "data/student_depression_dataset.csv"
    progress_tracker.update_detail("檢查資料檔案存在性")
    
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
        print(f"  ✅ 資料檔案存在 ({file_size:.1f} MB)")
        result = True
    else:
        print(f"  ❌ 找不到資料檔案: {data_path}")
        result = False
    
    progress_tracker.finish_test(result)
    return result

def test_font_loading(progress_tracker):
    """測試字體載入功能"""
    progress_tracker.next_test()
    
    try:
        progress_tracker.update_detail("載入字型模組")
        from src.font_loader import download_font_if_not_exist
        
        progress_tracker.update_detail("下載中文字型")
        font_path = download_font_if_not_exist()
        
        if font_path and os.path.exists(font_path):
            print(f"  ✅ 字體載入成功: {os.path.basename(font_path)}")
            result = True
        else:
            print(f"  ❌ 字體載入失敗")
            result = False
    except Exception as e:
        print(f"  ❌ 字體載入異常: {e}")
        result = False
    
    progress_tracker.finish_test(result)
    return result

def test_basic_functionality(progress_tracker):
    """測試基本功能"""
    progress_tracker.next_test()
    
    try:
        # 測試資料載入
        progress_tracker.update_detail("測試資料載入功能")
        from src.preprocess import load_data
        data_path = "data/student_depression_dataset.csv"
        
        if not os.path.exists(data_path):
            print("  ⚠️ 跳過功能測試 (無資料檔案)")
            result = True
            progress_tracker.finish_test(result)
            return result
            
        df = load_data(data_path)
        if len(df) > 0:
            print(f"  ✅ 資料載入成功 ({len(df)} 筆記錄)")
        else:
            print(f"  ❌ 資料載入失敗 (空資料集)")
            result = False
            progress_tracker.finish_test(result)
            return result
            
        # 測試完整預處理流程
        progress_tracker.update_detail("測試資料預處理功能")
        from src.preprocess import preprocess
        
        test_df = preprocess(data_path)
        print(f"  ✅ 完整預處理後: {len(test_df)} 筆資料")
        
        if len(test_df) > 100:
            print(f"  ✅ 資料預處理成功")
            
            # 測試模型訓練功能
            progress_tracker.update_detail("測試模型訓練功能")
            from src.model_utils import train_logistic_regression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            features = ['Academic Pressure_Value', 'Age', 'CGPA']
            available_features = [f for f in features if f in test_df.columns]
            
            if len(available_features) > 0:
                X = test_df[available_features].fillna(0)
                y = test_df['Depression']
                
                if len(X) > 50 and len(y) > 50:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    model = train_logistic_regression(X_train_scaled, y_train)
                    print(f"  ✅ 模型訓練成功")
                else:
                    print(f"  ⚠️ 可用資料不足，跳過模型測試")
            else:
                print(f"  ⚠️ 找不到必要特徵，跳過模型測試")
        else:
            print(f"  ⚠️ 預處理後資料不足，但功能正常")
            
        print(f"  ✅ 基本功能測試完成")
        result = True
        
    except Exception as e:
        print(f"  ❌ 功能測試異常: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        result = False
    
    progress_tracker.finish_test(result)
    return result

def main():
    """主測試函數"""
    # 初始化進度追蹤器
    progress_tracker = TestProgressTracker()
    progress_tracker.start()
    
    test_functions = [
        test_dependencies,
        test_imports,
        test_data_file,
        test_font_loading,
        test_basic_functionality
    ]
    
    try:
        # 執行所有測試
        for test_func in test_functions:
            test_func(progress_tracker)
        
        # 完成測試
        progress_tracker.complete()
        
        # 計算通過率
        passed = sum(1 for _, result in progress_tracker.results if result)
        total = len(progress_tracker.results)
        
        if passed == total:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷了測試程序")
        progress_tracker.complete()
        return 1
    except Exception as e:
        print(f"\n\n❌ 測試過程中發生異常: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        progress_tracker.complete()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 