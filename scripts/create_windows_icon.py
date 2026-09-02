"""从项目 PNG 图标生成 PyInstaller 在 Windows 下使用的 ICO 文件。"""

from pathlib import Path

from PIL import Image


root = Path(__file__).resolve().parents[1]
source = root / "resources" / "academic_agent_icon.png"
target = root / "resources" / "academic_agent_icon.ico"

if not source.is_file():
    raise SystemExit(f"找不到图标文件：{source}")

with Image.open(source) as image:
    image.convert("RGBA").save(
        target,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )

print(f"已生成 Windows 图标：{target}")
