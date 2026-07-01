"""Worker: 后台推理线程 (单管道直写 + 断点续传)"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal
from video_io.reader import VideoReader
from video_io.writer import VideoWriter

logger = logging.getLogger(__name__)


def _find_dovi_tool() -> str | None:
    import shutil
    from utils.config import APP_ROOT
    for n in ("dovi_tool", "dovi_tool.exe"):
        p = APP_ROOT / n
        if p.exists():
            return str(p)
    return shutil.which("dovi_tool") or shutil.which("dovi_tool.exe")


def _inject_dovi_rpu(output_path: str, src_rpu: str, audio_src: str):
    """后处理: dovi_tool inject-rpu 注入源 RPU (社区标准方案)。
    inject-rpu 自动处理帧数差异, 无需手动 duplicate。"""
    dovi = _find_dovi_tool()
    if not dovi:
        return
    from utils.config import find_ffmpeg
    ff = find_ffmpeg()
    pk = {}
    if sys.platform == "win32":
        pk["creationflags"] = subprocess.CREATE_NO_WINDOW

    logger.info("DoVi: 注入 RPU 到输出视频...")
    out_path = Path(output_path)
    tmp_hevc = str(out_path.parent / f"_{out_path.stem}_tmp.hevc")
    tmp_dovi = str(out_path.parent / f"_{out_path.stem}_tmp_dovi.hevc")
    tmp_mkv = str(out_path.parent / f"_{out_path.stem}_tmp_dovi.mkv")

    # 1. 提取输出 HEVC
    r = subprocess.run([ff, "-y", "-i", output_path, "-c:v", "copy",
                        "-bsf:v", "hevc_mp4toannexb", "-f", "hevc", tmp_hevc],
                       capture_output=True, timeout=300, **pk)
    if r.returncode != 0:
        logger.warning("DoVi: HEVC 提取失败")
        return

    # 2. inject-rpu
    r = subprocess.run([dovi, "inject-rpu", "-i", tmp_hevc,
                        "--rpu-in", src_rpu, "-o", tmp_dovi],
                       capture_output=True, timeout=120, **pk)
    os.unlink(tmp_hevc)
    if r.returncode != 0:
        logger.warning("DoVi: RPU 注入失败: %s",
                       r.stderr.decode(errors="replace")[-200:])
        try:
            os.unlink(tmp_dovi)
        except Exception:
            pass
        return

    # 3. 提取源字幕 (PGS/SRT 等, 跳过 DOVI side_data)
    tmp_subs = str(out_path.parent / f"_{out_path.stem}_subs.mkv")
    r = subprocess.run([ff, "-y", "-i", audio_src, "-map", "0:s?",
                        "-c:s", "copy", tmp_subs],
                       capture_output=True, timeout=60, **pk)
    has_subs = r.returncode == 0 and os.path.getsize(tmp_subs) > 1024
    if not has_subs:
        try: os.unlink(tmp_subs)
        except Exception: pass
        tmp_subs = None

    # 4. remux DoVi HEVC + 音频 + 字幕
    cmd = [ff, "-y", "-i", tmp_dovi, "-i", audio_src,
           "-map_metadata", "-1", "-map", "0:v", "-c:v", "copy",
           "-map", "1:a?", "-c:a", "copy"]
    if tmp_subs:
        cmd += ["-i", tmp_subs, "-map", "2:s?", "-c:s", "copy"]
    cmd.append(tmp_mkv)
    r = subprocess.run(cmd, capture_output=True, timeout=300, **pk)
    os.unlink(tmp_dovi)
    if tmp_subs:
        os.unlink(tmp_subs)
    if r.returncode != 0:
        logger.warning("DoVi: remux 失败: %s",
                       r.stderr.decode(errors="replace")[-300:])
        return

    # 4. 替换
    os.unlink(output_path)
    os.replace(tmp_mkv, output_path)
    logger.info("DoVi: RPU 注入完成 → %s", out_path.name)


class InterpolationWorker(QObject):
    progress = pyqtSignal(int, int)
    file_started = pyqtSignal(int)
    file_progress = pyqtSignal(int, int, int)
    file_finished = pyqtSignal(int, str)
    file_error = pyqtSignal(int, str)
    file_resumed = pyqtSignal(int, int)
    all_finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, file_list, engine, fps_multiplier=2,
                 encoder="libx264", crf=18, pix_fmt="yuv420p"):
        super().__init__()
        self.file_list = file_list
        self.engine = engine
        self.fps_multiplier = fps_multiplier
        self.encoder = encoder
        self.crf = crf
        self.pix_fmt = pix_fmt
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for vid, inp, out in self.file_list:
            if self._cancelled:
                break
            self.file_started.emit(vid)
            self._process_one(vid, inp, out)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if self._cancelled:
            self.cancelled.emit()
        self.all_finished.emit()

    def _process_one(self, vid, inp, out):
        try:
            reader = VideoReader(inp)
            out_path = Path(out)
            audio_src = inp
            _has_dovi = reader.has_dovi
            _vfr = reader.vfr_needs_cfr

            if _vfr:
                from video_io.reader import convert_vfr_to_cfr
                cfr_path = convert_vfr_to_cfr(inp, reader.fps,
                                              str(out_path.parent))
                reader.close()
                reader = VideoReader(cfr_path)
                reader._cfr_path = cfr_path

            fps_out = reader.fps * self.fps_multiplier

            # DoVi: 提取源 RPU (后处理注入)
            dovi_src_rpu = None
            if _has_dovi:
                d = _find_dovi_tool()
                if d:
                    dpk = {}
                    if sys.platform == "win32":
                        dpk["creationflags"] = subprocess.CREATE_NO_WINDOW
                    dovi_src_rpu = os.path.join(
                        str(out_path.parent),
                        f"{out_path.stem}_dovi_rpu.bin")
                    r = subprocess.run(
                        [d, "extract-rpu", inp, "-o", dovi_src_rpu],
                        capture_output=True, timeout=300, **dpk)
                    if r.returncode != 0:
                        logger.warning("DoVi: RPU 提取失败")
                        dovi_src_rpu = None
                    else:
                        logger.info("DoVi: 源 RPU 已提取")

            # 续传
            progress_file = out_path.parent / f"{out_path.stem}_progress.txt"
            skip_frames = 0
            if progress_file.exists():
                try:
                    skip_frames = int(progress_file.read_text(
                        encoding="utf-8").strip())
                    logger.info("续传: 跳过%d帧", skip_frames)
                    self.file_resumed.emit(vid, skip_frames)
                except Exception:
                    pass

            total_out = reader.total_frames * self.fps_multiplier

            writer = None
            prev = None
            out_idx = skip_frames
            in_idx = skip_frames // self.fps_multiplier

            for _ in range(in_idx):
                try:
                    prev = next(iter(reader))
                except StopIteration:
                    break

            t0 = time.perf_counter()
            for frame in reader:
                if self._cancelled:
                    reader.close()
                    if writer:
                        writer.close()
                    return

                if prev is not None:
                    mids = self.engine.interpolate_multi(
                        prev, frame, self.fps_multiplier)
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = VideoWriter(
                            out, fps_out, w, h, audio_src,
                            self.encoder, self.crf, self.pix_fmt,
                            skip_frames=skip_frames,
                            src_pix_fmt=getattr(reader, 'pix_fmt', 'yuv420p'),
                            src_sar=getattr(reader, 'sar', None),
                            color_space=getattr(reader, 'color_space', 'bt709'),
                            color_transfer=getattr(reader, 'color_transfer', 'bt709'),
                            color_primaries=getattr(reader, 'color_primaries', 'bt709'))
                        writer.write(prev)
                        out_idx += 1
                    for m in mids:
                        if self._cancelled:
                            reader.close()
                            writer.close()
                            return
                        writer.write(m)
                        out_idx += 1
                        self.progress.emit(out_idx, total_out)
                        self.file_progress.emit(vid, out_idx, total_out)
                    writer.write(frame)
                    out_idx += 1
                    self.progress.emit(out_idx, total_out)
                    self.file_progress.emit(vid, out_idx, total_out)
                prev = frame
                in_idx += 1

            reader.close()
            if writer is None and prev is not None:
                h, w = prev.shape[:2]
                writer = VideoWriter(out, fps_out, w, h, audio_src,
                                     self.encoder, self.crf, self.pix_fmt,
                                     src_pix_fmt=getattr(reader, 'pix_fmt', 'yuv420p'),
                                     src_sar=getattr(reader, 'sar', None),
                                     color_space=getattr(reader, 'color_space', 'bt709'),
                                     color_transfer=getattr(reader, 'color_transfer', 'bt709'),
                                     color_primaries=getattr(reader, 'color_primaries', 'bt709'))
                writer.write(prev)
            if writer:
                writer.close()
                if self._cancelled:
                    return

            # DoVi 后处理注入
            if dovi_src_rpu and not self._cancelled:
                _inject_dovi_rpu(out, dovi_src_rpu, audio_src)

            elapsed = time.perf_counter() - t0
            logger.info("完成: %s (%d帧, %.1fs)",
                        Path(inp).name, out_idx, elapsed)
            self.file_finished.emit(vid, out)

            # 清理临时文件
            if getattr(reader, '_cfr_path', None):
                try:
                    os.unlink(reader._cfr_path)
                except Exception:
                    pass
            if dovi_src_rpu:
                try:
                    os.unlink(dovi_src_rpu)
                except Exception:
                    pass

        except Exception as e:
            logger.exception("失败: %s", inp)
            self.file_error.emit(vid, str(e))
