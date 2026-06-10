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

    m = re.search(r"Stream #\d+:\d+.*?Video:.*?,\s+(\d+)x(\d+).*?,\s+([\d.]+)\s+fps", info)
    if not m:
        raise RuntimeError(f"未找到视频流: {video_path}")
    width, height = int(m.group(1)), int(m.group(2))
    fps = float(m.group(3))
    total_frames = int(duration_s * fps)

    offset = 0.0
    m = re.search(r"Stream #\d+:\d+.*?Video:.*?,\s+start\s+([\d.]+)", info)
    if m:
        offset = float(m.group(1))

    return {"width": width, "height": height, "fps": fps,
            "total_frames": total_frames, "duration": duration_s,
            "start_time": offset}


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
        logger.info("探测: %dx%d %.3ffps %d帧 (偏移 %.3fs)",
                     self.width, self.height, self.fps,
                     self.total_frames, self.start_time)

        ff = find_ffmpeg()
        popen_kw = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            [ff, "-i", self.path,
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
