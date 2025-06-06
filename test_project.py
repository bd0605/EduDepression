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

def test_imports():
    """測試所有模組匯入"""
    print("🔍 測試模組匯入...")
    
    modules_to_test = [
        "src.preprocess",
        "src.plot_utils", 
        "src.model_utils",
        "src.db_utils",
        "src.font_loader"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"  ⚠️ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_data_file():
    """檢查資料檔案是否存在"""
    print("\n📂 檢查資料檔案...")
    
    data_path = "data/student_depression_dataset.csv"
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
        print(f"  ✅ 資料檔案存在 ({file_size:.1f} MB)")
        return True
    else:
        print(f"  ❌ 找不到資料檔案: {data_path}")
        return False

def test_font_loading():
    """測試字體載入功能"""
    print("\n🔤 測試字體載入...")
    
    try:
        from src.font_loader import download_font_if_not_exist
        font_path = download_font_if_not_exist()
        if font_path and os.path.exists(font_path):
            print(f"  ✅ 字體載入成功: {os.path.basename(font_path)}")
            return True
        else:
            print(f"  ❌ 字體載入失敗")
            return False
    except Exception as e:
        print(f"  ❌ 字體載入異常: {e}")
        return False

def test_basic_functionality():
    """測試基本功能"""
    print("\n⚙️ 測試基本功能...")
    
    try:
        # 測試資料載入
        from src.preprocess import load_data
        data_path = "data/student_depression_dataset.csv"
        
        if not os.path.exists(data_path):
            print("  ⚠️ 跳過功能測試 (無資料檔案)")
            return True
            
        df = load_data(data_path)
        if len(df) > 0:
            print(f"  ✅ 資料載入成功 ({len(df)} 筆記錄)")
        else:
            print(f"  ❌ 資料載入失敗 (空資料集)")
            return False
            
        # 測試完整預處理流程
        from src.preprocess import preprocess
        print(f"    測試完整預處理流程...")
        
        # 使用較多資料來測試完整流程
        test_df = preprocess(data_path)
        print(f"    完整預處理後: {len(test_df)} 筆資料")
        
        if len(test_df) > 100:
            print(f"  ✅ 資料預處理成功")
            
            # 測試模型訓練功能
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
        return True
        
    except Exception as e:
        print(f"  ❌ 功能測試異常: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """檢查依賴套件"""
    print("\n📦 檢查依賴套件...")
    
    required_packages = [
        "pandas", "numpy", "scipy", "matplotlib", 
        "seaborn", "sklearn", "sqlalchemy", "pymysql", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
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
        print(f"\n💡 請安裝缺少的套件: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """主測試函數"""
    print("🚀 EduDepression 專案完整性測試")
    print("=" * 50)
    
    tests = [
        ("依賴套件", test_dependencies),
        ("模組匯入", test_imports),
        ("資料檔案", test_data_file),
        ("字體載入", test_font_loading),
        ("基本功能", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"測試 '{test_name}' 發生異常: {e}")
            results.append((test_name, False))
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試結果總結:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 總計: {passed}/{len(results)} 項測試通過")
    
    if passed == len(results):
        print("🎉 所有測試通過！專案設定正確。")
        return 0
    else:
        print("⚠️ 部分測試失敗，請檢查上述錯誤訊息。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 