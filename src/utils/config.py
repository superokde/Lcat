"""路径管理"""
import sys
from pathlib import Path

# 便携 Python: 项目根 = main.py 的祖父目录 (Lcat/)
APP_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR_OFFICIAL = APP_ROOT / "models" / "rife_official"
MODEL_DIR_CUSTOM = APP_ROOT / "models" / "rife_custom"

DEFAULT_MODEL = "v4.25"
DEFAULT_FPS_MULTIPLIER = 2
DEFAULT_ENCODER = "libx264"
DEFAULT_CRF = 18
SUPPORTED_INPUT_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


def get_model_version(dirname: str) -> str | None:
    import re
    m = re.search(r"v(\d+)[._](\d+)", dirname, re.IGNORECASE)
    return f"v{m.group(1)}.{m.group(2)}" if m else None


def find_ffmpeg() -> str:
    import shutil
    # 项目同目录
    for n in ("ffmpeg", "ffmpeg.exe"):
        p = APP_ROOT / n
        if p.exists():
            return str(p)
    # 系统 PATH
    f = shutil.which("ffmpeg")
    if f:
        return f
    raise FileNotFoundError("ffmpeg 未找到，请放入 Lcat 目录或系统 PATH")
