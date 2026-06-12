"""ffmpeg 单管道直写 — 无分段，GPU 利用率最高。

输出视频 + 音频/字幕保留 + 色彩元数据 + PTS 偏移补偿。
断点续传: 外部记录已处理帧数，重跑时跳过帧。
"""
import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path
import numpy as np

from utils.config import find_ffmpeg

logger = logging.getLogger(__name__)

_MIN_FREE_DISK_MB = 500


def _fps_to_rational(fps: float, tv_compat: bool = True) -> str:
    """浮点帧率转有理数。tv_compat=True 时输出电视兼容的标准帧率。

    音画同步分析 (tv_compat 模式):
    - 23.98→24(+0.08%), 29.97→30(+0.1%): 一部电影差几秒，无法感知
    - 视频帧数不变，只改变播放速度元数据，音轨同比例加速
    - 与 YouTube/Netflix 25fps→24fps 转换属于同级别处理"""

    # 电视兼容：非标帧率向上取整到标准值
    if tv_compat:
        TV_MAP = {
            23.976: "24/1", 23.98: "24/1",
            29.97: "30/1",
            47.95: "48/1", 47.96: "48/1",
            59.94: "60/1",
            71.93: "72/1", 71.94: "72/1",
            95.90: "96/1", 95.92: "96/1",
            119.88: "120/1",
            143.86: "144/1", 143.88: "144/1",
            191.81: "192/1", 191.84: "192/1",
        }
        for f_val, rat in TV_MAP.items():
            if abs(fps - f_val) < 0.02:
                return rat

    # 精确模式或回退
    NTSC = 1001
    COMMON = {
        23.976: 24000, 23.98: 24000,
        29.97: 30000,
        47.95: 48000, 47.96: 48000,
        59.94: 60000,
        71.928: 72000, 71.93: 72000,
        95.904: 96000, 95.92: 96000,
        119.88: 120000,
        143.856: 144000, 143.86: 144000,
        191.808: 192000, 191.84: 192000,
    }
    for f_val, num in COMMON.items():
        if abs(fps - f_val) < 0.02:
            return f"{num}/{NTSC}"

    for base in [24, 25, 30, 48, 50, 60, 72, 75, 96, 100, 120, 144, 150, 192, 200, 240]:
        if abs(fps - base) < 0.01:
            return f"{base}/1"

    num = round(fps * 1000)
    a, b = num, 1000
    while b:
        a, b = b, a % b
    return f"{num // a}/{1000 // a}"


def _encode_params(encoder: str, crf: int, pix_fmt: str) -> list:
    """返回编码器参数列表。"""
    params = []
    if "av1_nvenc" in encoder:
        params += ["-qp", str(crf), "-preset", "p7", "-tune", "hq",
                   "-rc-lookahead", "32", "-spatial_aq", "1",
                   "-temporal_aq", "1", "-aq-strength", "8"]
    elif "nvenc" in encoder:
        params += ["-rc", "vbr_hq", "-cq", str(crf), "-b:v", "0",
                   "-preset", "p7", "-tune", "hq",
                   "-bf:v", "3", "-b_ref_mode", "middle",
                   "-rc-lookahead", "32",
                   "-spatial_aq", "1", "-temporal_aq", "1",
                   "-aq-strength", "8"]
    elif encoder == "libsvtav1":
        params += ["-crf", str(crf), "-preset", "6"]
    else:
        params += ["-crf", str(crf)]
        if encoder == "libx264":
            params += ["-preset", "slow", "-bf", "3",
                       "-b_strategy", "2", "-refs", "6"]
        elif encoder == "libx265":
            params += ["-preset", "slow", "-bf", "3",
                       "-x265-params", "aq-mode=3"]
    # 始终指定输出像素格式: 防止 libx265 接收 BGR 输入后自动选 gbrp/Rext
    # → HEVC Rext 格式所有消费设备无法解码 (小米电视/当贝/VLC 均不支持)
    if pix_fmt in ("yuv420p10le", "p010le"):
        params += ["-pix_fmt", pix_fmt]
    else:
        params += ["-pix_fmt", "yuv420p"]
    return params


def _nvenc_supports_10bit(encoder: str) -> bool:
    try:
        kw = {"capture_output": True, "timeout": 15}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        r = subprocess.run(
            [find_ffmpeg(), "-y", "-f", "lavfi", "-i",
             "color=black:s=320x256:r=1:d=3",
             "-c:v", encoder, "-pix_fmt", "p010le",
             "-frames:v", "1", "-f", "mp4", "NUL"],
            **kw)
        return r.returncode == 0
    except Exception:
        return False


def _run_ffmpeg(cmd: list, timeout: int = 3600) -> subprocess.CompletedProcess:
    kw = {"capture_output": True, "timeout": timeout}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(cmd, **kw)


class VideoWriter:
    """单管道直写器。帧通过 stdin 送入 ffmpeg，直接输出最终视频。"""

    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 audio_src: str = None, encoder: str = "libx264",
                 crf: int = 18, pix_fmt: str = "yuv420p",
                 skip_frames: int = 0, src_pix_fmt: str = "yuv420p",
                 src_sar: str = None):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.encoder = encoder
        self.crf = crf
        self.pix_fmt = pix_fmt
        self.skip_frames = skip_frames
        self._frame_count = 0
        self._written = 0
        self._proc = None
        self._stderr_thread = None

        # 10bit 自动降级: 源是 8bit → 输出用 8bit
        if src_pix_fmt and "10" not in src_pix_fmt and pix_fmt in ("yuv420p10le", "p010le"):
            logger.info("源视频为 8bit (%s)，10bit 编码降级为 8bit", src_pix_fmt)
            self.pix_fmt = "yuv420p"

        self._fps_exact = _fps_to_rational(fps, tv_compat=False)
        # 计算匹配帧率的 timebase (Doom9 社区方案: video_track_timescale)
        self._timescale = int(self._fps_exact.split("/")[0])
        self._src_sar = src_sar

        # 磁盘预检
        free_mb = shutil.disk_usage(self.output_path.parent).free // (1024 * 1024)
        if free_mb < _MIN_FREE_DISK_MB:
            raise RuntimeError(f"磁盘空间不足: {free_mb}MB 剩余 (需>={_MIN_FREE_DISK_MB}MB)")

        # 探测音频源 PTS 偏移
        offset = 0.0
        if audio_src:
            try:
                from video_io.reader import _probe
                meta = _probe(audio_src)
                if meta["start_time"] > 0.05:
                    offset = meta["start_time"]
                    logger.info("检测到视频延迟 %.3fs，已补偿", offset)
            except Exception:
                pass

        self._start_writer(audio_src, offset)

    def _start_writer(self, audio_src: str = None, offset: float = 0.0):
        ff = find_ffmpeg()
        ext = self.output_path.suffix.lower()

        cmd = [ff, "-y"]

        # 视频输入 (stdin raw BGR)
        if offset > 0.05:
            cmd += ["-itsoffset", str(offset)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", self._fps_exact,
                "-i", "-"]

        # 音频/字幕输入
        if audio_src:
            cmd += ["-i", audio_src]

        # 视频编码
        cmd += ["-map_metadata", "-1", "-map", "0:v"]
        cmd += ["-c:v", self.encoder]
        cmd += _encode_params(self.encoder, self.crf, self.pix_fmt)
        # 恒定帧率 + 精确时间戳 (Doom9 社区方案)
        cmd += ["-vsync", "cfr"]
        cmd += ["-video_track_timescale", str(self._timescale)]

        # 10bit 降级检测
        if self.pix_fmt in ("yuv420p10le", "p010le"):
            if "nvenc" in self.encoder and not _nvenc_supports_10bit(self.encoder):
                logger.warning("GPU 不支持 10bit，降级为 8bit")

        # 音频/字幕复制
        if audio_src:
            cmd += ["-map", "1:a?", "-map", "1:s?"]
            cmd += ["-c:a", "copy"]
            cmd += ["-c:s", "mov_text"] if ext in (".mp4", ".m4v") else ["-c:s", "copy"]
            cmd += ["-map_metadata:s", "-1"]

        # SAR 保留 (修复 DVD 源 SAR 丢失问题)
        if self._src_sar:
            cmd += ["-sar", self._src_sar]

        # 色彩元数据 (电视兼容)
        cmd += ["-color_range", "1", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-colorspace", "bt709"]

        cmd.append(str(self.output_path))

        popen_kw = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL,
                     "stderr": subprocess.PIPE}
        if sys.platform == "win32":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._proc = subprocess.Popen(cmd, **popen_kw)
        self._stderr_chunks = []

        def _drain_stderr():
            for chunk in iter(self._proc.stderr.readline, b""):
                self._stderr_chunks.append(chunk)

        self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()

        logger.info("ffmpeg 管道写入: %dx%d %.2ffps %s",
                     self.width, self.height, self.fps, self.output_path.name)

    def write(self, frame: np.ndarray):
        """写入一帧 (BGR HWC uint8)。"""
        if self._proc is None:
            raise RuntimeError("写入器已关闭")
        if self._frame_count < self.skip_frames:
            self._frame_count += 1
            return
        if frame.shape[:2] != (self.height, self.width):
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))
        self._proc.stdin.write(frame.tobytes())
        self._frame_count += 1
        self._written += 1

    def close(self) -> int:
        """关闭管道，等待 ffmpeg 完成。返回输出帧数。"""
        if self._proc is None:
            return self._written
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._stderr_thread.join(timeout=10)
        self._proc.wait(timeout=3600)
        if self._proc.returncode != 0:
            stderr_text = b"".join(self._stderr_chunks).decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"编码失败 (rc={self._proc.returncode}): {stderr_text}")
        if (not self.output_path.exists() or
                self.output_path.stat().st_size < 1024):
            raise RuntimeError(f"输出无效: {self.output_path}")
        logger.info("完成: %s (%d帧)", self.output_path.name, self._written)
        return self._written

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
