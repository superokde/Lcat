"""ffmpeg 管道视频读取 (BGR 格式)"""
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

from utils.config import find_ffmpeg

logger = logging.getLogger(__name__)


def _probe(video_path: str) -> dict:
    ff = find_ffmpeg()
    run_kw = {"capture_output": True, "timeout": 30}
    if sys.platform == "win32":
        run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    r = subprocess.run([ff, "-i", video_path], **run_kw)
    info = r.stderr.decode("utf-8", errors="replace")

    m = re.search(r"Duration:\s+(\d+):(\d+):(\d+)\.(\d+)", info)
    if not m:
        raise RuntimeError(f"无法解析视频信息: {video_path}")
    duration_s = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 +
                  int(m.group(3)) + int(m.group(4)) / (10 ** len(m.group(4))))

    m = re.search(r"Stream #\d+:\d+.*?Video:\s+(\S+).*?,\s+(\d+)x(\d+)(?:[^,]*?),\s+([\d.]+)\s+fps", info)
    if not m:
        m = re.search(r"Stream #\d+:\d+.*?Video:.*?,\s+(\d+)x(\d+).*?,\s+([\d.]+)\s+fps", info)
        if not m:
            raise RuntimeError(f"未找到视频流: {video_path}")
        codec_pix = ""
    else:
        codec_pix = m.group(1)

    if codec_pix:
        width, height = int(m.group(2)), int(m.group(3))
    else:
        width, height = int(m.group(1)), int(m.group(2))
    fps_ffmpeg = float(m.group(4 if codec_pix else 3))

    # VFR 检测: 用 ffprobe 获取精确 avg_frame_rate (零额外耗时)
    run2_kw = {"capture_output": True, "timeout": 15}
    if sys.platform == "win32":
        run2_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    # 查找 ffprobe: 项目同目录优先，其次系统 PATH
    ff_dir = os.path.dirname(ff)
    ffprobe_bin = os.path.join(ff_dir, "ffprobe.exe")
    if not os.path.isfile(ffprobe_bin):
        import shutil
        ffprobe_bin = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
        if not ffprobe_bin:
            raise RuntimeError("ffprobe 未找到，请将 ffprobe.exe 放入项目目录或系统 PATH")
    r2 = subprocess.run(
        [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,color_space,color_transfer,color_primaries",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        **run2_kw)
    out2 = r2.stdout.decode("utf-8", errors="replace")
    lines = [l.strip() for l in out2.splitlines() if l.strip()]

    # 分离帧率(含"/")和色彩字段
    fps_avg = fps_ffmpeg
    color_vals = []
    for line in lines:
        if "/" in line and not line.startswith("bt"):
            try:
                num, den = line.split("/")
                fps_avg = int(num) / int(den) if int(den) > 0 else fps_ffmpeg
            except (ValueError, ZeroDivisionError):
                pass
        elif line.startswith("bt") or line in ("smpte2084", "arib-std-b67",
                "smpte170m", "smpte240m", "iec61966-2-4", "linear"):
            color_vals.append(line)

    # ffprobe default writer 输出顺序不可控，按典型 HDR 字段特征分类
    color_space = "bt709"
    color_transfer = "bt709"
    color_primaries = "bt709"
    for v in color_vals:
        if "nc" in v or "ncl" in v:
            color_space = v
        elif v in ("smpte2084", "arib-std-b67", "smpte428", "linear", "bt470bg",
                   "bt470m", "smpte170m", "smpte240m", "iec61966-2-4", "bt709",
                   "bt1361e", "log100", "log316", "iec61966-2-1", "bt2020-10",
                   "bt2020-12", "smpte428-1"):
            color_transfer = v
        elif v.startswith("bt") or v.startswith("smpte"):
            color_primaries = v
    # HDR 同义修正
    if color_transfer == "smpte2084":
        if "bt2020" not in color_primaries:
            color_primaries = "bt2020"
        if "bt2020" not in color_space:
            color_space = "bt2020nc"

    # VFR 判定: 比对容器级 r_frame_rate 和流级 avg_frame_rate
    r_frame_rate = None
    m_rfr = re.search(r"Stream #\d+:\d+.*?Video:.*?\b([\d.]+)\s+fps", info)
    # 也尝试从 ffprobe 获取 r_frame_rate
    r3 = subprocess.run(
        [ff, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        **run2_kw)
    rfr_str = r3.stdout.decode("utf-8", errors="replace").strip()
    try:
        num2, den2 = rfr_str.split("/")
        r_frame_rate_val = int(num2) / int(den2) if int(den2) > 0 else 0
    except (ValueError, ZeroDivisionError):
        r_frame_rate_val = fps_ffmpeg

    fps = fps_avg
    vfr_needs_cfr = False
    if fps_avg > 0 and r_frame_rate_val > 0:
        diff = abs(r_frame_rate_val - fps_avg) / fps_avg
        if diff > 0.05:
            fps = fps_avg
            vfr_needs_cfr = True
            logger.warning("VFR 检测: 容器 %.3f, 实际 %.3f (差异 %.0f%%)",
                           r_frame_rate_val, fps_avg, diff * 100)

    total_frames = int(duration_s * fps)

    # 提取像素格式
    pix_fmt = "yuv420p"
    m_pix = re.search(r"(yuv\w+)\s*,?\s*\d+x\d+", info)
    if not m_pix:
        m_pix = re.search(r"Video:\s+\S+\s+\([^)]*,\s+(yuv\w+)", info)
    if not m_pix:
        m_pix = re.search(r"Video:\s+\S+\s+\([^)]*\),\s+(yuv\w+)", info)
    if m_pix:
        pix_fmt = m_pix.group(1)

    # 提取 SAR
    sar = None
    m_sar = re.search(r"SAR\s+(\d+:\d+)", info)
    if m_sar:
        sar = m_sar.group(1)

    offset = 0.0
    m = re.search(r"Stream #\d+:\d+.*?Video:.*?,\s+start\s+([\d.]+)", info)
    if m:
        offset = float(m.group(1))

    return {"width": width, "height": height, "fps": fps,
            "total_frames": total_frames, "duration": duration_s,
            "start_time": offset, "pix_fmt": pix_fmt, "sar": sar,
            "vfr_warn": vfr_needs_cfr,
            "color_space": color_space, "color_transfer": color_transfer,
            "color_primaries": color_primaries}


def convert_vfr_to_cfr(src_path: str, fps_target: float, work_dir: str) -> str:
    """VFR → CFR 预处理。返回临时 CFR 文件路径。使用 ultrafast 编码最小化时间损耗。"""
    import tempfile
    ff = find_ffmpeg()
    cfr_path = os.path.join(work_dir, f"{Path(src_path).stem}_cfr_temp.mkv")
    logger.info("VFR→CFR 预处理: %s → %.3ffps", Path(src_path).name, fps_target)
    run_kw = {"capture_output": True, "timeout": 7200}
    if sys.platform == "win32":
        run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    r = subprocess.run(
        [ff, "-fflags", "+genpts", "-i", src_path,
         "-vf", f"fps=fps={fps_target}",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
         "-an", "-sn", cfr_path],
        **run_kw)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(f"VFR→CFR 转换失败: {err}")
    logger.info("VFR→CFR 完成: %s", cfr_path)
    return cfr_path


class VideoReader:
    """ffmpeg 管道读取器，逐帧 yield BGR HWC uint8 数组。"""

    def __init__(self, video_path: str | Path):
        self.path = str(video_path)
        meta = _probe(self.path)
        self.width = meta["width"]
        self.height = meta["height"]
        self.fps = meta["fps"]
        self.total_frames = meta["total_frames"]
        self.start_time = meta["start_time"]
        self.pix_fmt = meta.get("pix_fmt", "yuv420p")
        self.sar = meta.get("sar")
        self.vfr_needs_cfr = meta.get("vfr_warn", False)
        self._cfr_path: str | None = None
        self.color_space = meta.get("color_space", "bt709")
        self.color_transfer = meta.get("color_transfer", "bt709")
        self.color_primaries = meta.get("color_primaries", "bt709")
        logger.info("探测: %dx%d %.3ffps %d帧 (偏移 %.3fs)",
                     self.width, self.height, self.fps,
                     self.total_frames, self.start_time)

        ff = find_ffmpeg()
        popen_kw = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            [ff, "-fflags", "+genpts", "-vsync", "0", "-i", self.path,
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-vcodec", "rawvideo",
             "-an", "-sn", "-"],
            **popen_kw)

        self._frame_bytes = self.width * self.height * 3
        self._pos = 0

    def __iter__(self):
        while True:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                self.height, self.width, 3).copy()
            self._pos += 1
            yield frame

        if self._pos != self.total_frames:
            logger.warning("实际帧数 %d != 探测 %d，以实际为准",
                           self._pos, self.total_frames)
            self.total_frames = self._pos

    def close(self):
        if self._proc:
            self._proc.stdout.close()
            self._proc.wait(timeout=10)
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
