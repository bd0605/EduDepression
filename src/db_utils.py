# -*- coding: utf-8 -*-
"""
MySQL 匯入匯出函式模組

此模組提供 EduDepression 專案的 MySQL 資料庫連接、查詢與匯入匯出功能，
支援 DataFrame 與 MySQL 間的雙向轉換操作。
"""

import pandas as pd
from urllib.parse import urlparse
from sqlalchemy import create_engine, text
import pymysql
import os
import sys

def get_engine(mysql_uri=None, verbose=False):
    """
    創建 SQLAlchemy 引擎連接 MySQL，
    並自動建立資料庫（若不存在）。
    """
    # 1. 準備連線字串
    if mysql_uri is None:
        mysql_uri = os.environ.get('MYSQL_URI',
            "mysql+pymysql://root@localhost:3306/depression_db?charset=utf8mb4"
        )
    if verbose:
        print(f"🔗 使用連線字串：{mysql_uri}")

    # 2. 先用不帶資料庫的 URI 連一次，確保資料庫存在
    parsed = urlparse(mysql_uri)
    db_name = parsed.path.lstrip('/')  # e.g. "depression_db"
    no_db_uri = mysql_uri.replace(f"/{db_name}", "/")
    try:
        tmp_engine = create_engine(no_db_uri)
        with tmp_engine.connect() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            ))
        if verbose:
            print(f"✅ 資料庫 `{db_name}` 已確保存在")
    except Exception as e:
        # 只警告，不中斷：如果這步失敗，後面還是會試著直接連
        print(f"⚠️ 自動建立資料庫 `{db_name}` 失敗：{e}")

    # 3. 再建立帶資料庫的 engine
    try:
        engine = create_engine(mysql_uri)
        return engine
    except Exception as e:
        print(f"❌ 無法建立資料庫引擎: {e}")
        return None


def test_connection(mysql_uri=None):
    engine = get_engine(mysql_uri, verbose=True)
    if engine is None:
        return False

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print(f"❌ 資料庫連接測試失敗: {e}")
        print("請檢查以下幾點：")
        print("1️⃣ MySQL 帳號密碼（例如 root/空密碼）")
        print("2️⃣ 是否啟動了 MySQL 伺服器")
        print("3️⃣ 是否已建立 depression_db 資料庫")
        print("🔧 phpMyAdmin 使用者建議 root@localhost、密碼留空")
        return False


def export_to_mysql(df, table_name="student_depression", mysql_uri=None, if_exists="replace"):
    """
    將 DataFrame 匯出至 MySQL 資料表

    Args:
        df (pandas.DataFrame): 要匯出的資料框
        table_name (str, optional): 目標資料表名稱，預設為 "student_depression"
        mysql_uri (str, optional): MySQL 連接字串，若為 None，則從 get_engine 取得。
        if_exists (str, optional): 若資料表已存在的處理方式，可能的值為 "fail", "replace", "append"，預設為 "replace"。
                                 - "fail": 若資料表已存在則拋出例外
                                 - "replace": 若資料表已存在則先刪除再重建
                                 - "append": 若資料表已存在則追加資料

    Returns:
        bool: 匯出是否成功
    """
    # 取得引擎
    engine = get_engine(mysql_uri)
    if engine is None:
        return False
    
    # 匯出資料
    try:
        df.to_sql(
            table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            chunksize=1000  # 分批寫入，避免一次寫入過多資料
        )
        print(f"成功匯出 {len(df)} 筆資料至 {table_name} 資料表")
        return True
    except Exception as e:
        print(f"資料匯出失敗: {e}")
        return False

def import_from_mysql(query, mysql_uri=None):
    """
    從 MySQL 匯入資料至 DataFrame

    Args:
        query (str): SQL 查詢語句
        mysql_uri (str, optional): MySQL 連接字串，若為 None，則從 get_engine 取得。

    Returns:
        pandas.DataFrame: 查詢結果資料框，若查詢失敗則回傳 None
    """
    # 取得引擎
    engine = get_engine(mysql_uri)
    if engine is None:
        return None
    
    # 匯入資料
    try:
        df = pd.read_sql(query, engine)
        print(f"成功匯入 {len(df)} 筆資料")
        return df
    except Exception as e:
        print(f"資料匯入失敗: {e}")
        return None

def create_schema(sql_file_path, mysql_uri=None):
    """
    執行 SQL 檔案以建立資料庫結構

    Args:
        sql_file_path (str): SQL 檔案路徑
        mysql_uri (str, optional): MySQL 連接字串，若為 None，則從 get_engine 取得。

    Returns:
        bool: 建立結構是否成功
    """
    # 取得引擎
    engine = get_engine(mysql_uri)
    if engine is None:
        return False
    
    # 讀取 SQL 檔案
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_script = file.read()
    except Exception as e:
        print(f"無法讀取 SQL 檔案: {e}")
        return False
    
    # 移除 USE 語句並依分號拆語句逐條執行
    lines = []
    for ln in sql_script.splitlines():
        # 跳過 USE database; 指令
        if ln.strip().upper().startswith('USE '):
            continue
        lines.append(ln)
    cleaned = "\n".join(lines)

    try:
        with engine.connect() as conn:
            # 按分號切割，每段非空才執行
            for stmt in cleaned.split(';'):
                stmt = stmt.strip()
                if not stmt:
                    continue
                conn.execute(text(stmt))
            conn.commit()
        print(f"✅ 成功執行 SQL 腳本: {sql_file_path}")
        return True
    except Exception as e:
        print(f"❌ SQL 腳本執行失敗: {e}")
        return False

def get_depression_by_pressure(mysql_uri=None):
    """
    取得各壓力層級的憂鬱比例，用於 Grafana 視覺化

    Args:
        mysql_uri (str, optional): MySQL 連接字串，若為 None，則從 get_engine 取得。

    Returns:
        pandas.DataFrame: 查詢結果資料框，包含壓力層級與憂鬱比例
    """
    # SQL 查詢語句：計算各壓力層級的憂鬱比例
    query = """
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
    """
    
    # 執行查詢
    return import_from_mysql(query, mysql_uri)

def get_pressure_by_degree(mysql_uri=None):
    """
    取得各學歷層級的學業壓力平均值，用於 Grafana 視覺化

    Args:
        mysql_uri (str, optional): MySQL 連接字串，若為 None，則從 get_engine 取得。

    Returns:
        pandas.DataFrame: 查詢結果資料框，包含學歷層級與平均學業壓力
    """
    # SQL 查詢語句：計算各學歷層級的平均學業壓力
    query = """
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
    """
    
    # 執行查詢
    return import_from_mysql(query, mysql_uri)

def get_pressure_stats_by_gender(mysql_uri=None):
    """
    取得各性別的學業壓力統計，用於 Grafana 視覺化

    Args:
        mysql_uri (str, optional): MySQL 連接字串，若為 None，則從 get_engine 取得。

    Returns:
        pandas.DataFrame: 查詢結果資料框，包含性別、學業壓力均值與標準差
    """
    # SQL 查詢語句：計算各性別的學業壓力統計
    query = """
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
    """
    
    # 執行查詢
    return import_from_mysql(query, mysql_uri)

# 當直接執行此模組時進行測試
if __name__ == "__main__":
    # 測試資料庫連接
    if test_connection():
        print("資料庫連接測試成功！")
    else:
        print("資料庫連接測試失敗！")
        sys.exit(1)
    
    # 測試建立結構
    sql_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "create_table.sql")
    if os.path.exists(sql_file_path):
        print("嘗試建立資料庫結構...")
        create_schema(sql_file_path)
    else:
        print(f"找不到 SQL 檔案: {sql_file_path}")
    
    # 測試資料上傳與查詢
    try:
        from preprocess import preprocess
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "student_depression_dataset.csv")
        df = preprocess(data_path)
        
        if export_to_mysql(df):
            print("資料匯出測試成功！")
            
            # 測試各種查詢
            print("\n各壓力層級的憂鬱比例:")
            print(get_depression_by_pressure())
            
            print("\n各學歷層級的學業壓力:")
            print(get_pressure_by_degree())
            
            print("\n各性別的學業壓力統計:")
            print(get_pressure_stats_by_gender())
        else:
            print("資料匯出測試失敗！")
    except Exception as e:
        print(f"資料處理或匯出過程發生錯誤: {e}")
