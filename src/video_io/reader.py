"""ffmpeg 管道视频读取 (BGR 格式)"""
import logging
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
    r2 = subprocess.run(
        [ff, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        **run2_kw)
    avg_str = r2.stdout.decode("utf-8", errors="replace").strip()
    try:
        num, den = avg_str.split("/")
        fps_avg = int(num) / int(den) if int(den) > 0 else fps_ffmpeg
    except (ValueError, ZeroDivisionError):
        fps_avg = fps_ffmpeg

    # VFR 判定: 流级平均帧率与容器声明差异 >5%
    fps = fps_avg
    vfr_warn = False
    if fps_avg > 0 and abs(fps_avg - fps_ffmpeg) / max(fps_avg, fps_ffmpeg) > 0.05:
        fps = fps_avg
        vfr_warn = True
        logger.warning("VFR 检测: 声明 %.3f, 实际 %.3f (差异 %.1f%%)，使用实际帧率",
                       fps_ffmpeg, fps_avg,
                       abs(fps_avg - fps_ffmpeg) / fps_avg * 100)

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
            "vfr_warn": vfr_warn}


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
        logger.info("探测: %dx%d %.3ffps %d帧 (偏移 %.3fs)",
                     self.width, self.height, self.fps,
                     self.total_frames, self.start_time)

        ff = find_ffmpeg()
        popen_kw = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            [ff, "-vsync", "0", "-i", self.path,
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
