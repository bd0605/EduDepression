import os
import requests
import platform
from pathlib import Path

def download_font_if_not_exist():
    system = platform.system()
    font_name = "NotoSansCJKjp-Regular.otf"

    # 根據作業系統設定儲存路徑
    if system == "Windows":
        font_dir = Path("fonts")  # 放在專案內，避免修改系統字體
    elif system == "Darwin":  # macOS
        font_dir = Path.home() / "Library/Fonts"
    elif system == "Linux":
        font_dir = Path.home() / ".fonts"
    else:
        raise OSError("不支援的作業系統")

    font_path = font_dir / font_name

    if not font_path.exists():
        print(f"📥 準備下載字體到：{font_path}")
        font_dir.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        try:
            with open(font_path, 'wb') as f:
                f.write(requests.get(url).content)
            print("✅ 字體下載完成！")
        except Exception as e:
            print(f"❌ 字體下載失敗：{e}")
            return None

    return str(font_path)
