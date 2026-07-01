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
from utils.config import find_ffmpeg

logger = logging.getLogger(__name__)


def _find_dovi_tool() -> str | None:
    """查找 dovi_tool: 项目同目录优先。"""
    import shutil
    from utils.config import APP_ROOT
    for n in ("dovi_tool", "dovi_tool.exe"):
        p = APP_ROOT / n
        if p.exists():
            return str(p)
    return shutil.which("dovi_tool") or shutil.which("dovi_tool.exe")


def _process_dovi_rpu(inp: str, total_source_frames: int,
                      total_output_frames: int, work_dir: str) -> str | None:
    """提取源 DoVi RPU → 生成匹配输出帧数的 RPU → 返回 RPU 文件路径。"""
    dovi = _find_dovi_tool()
    if not dovi:
        logger.warning("dovi_tool 未找到，跳过 DoVi RPU 处理")
        return None

    ff = find_ffmpeg()
    popen_kw = {}
    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW

    # 1. 提取源 HEVC 流 (stream copy, ~3-5s)
    src_hevc = os.path.join(work_dir, "_dovi_src.hevc")
    logger.info("DoVi: 提取源 HEVC 流...")
    r = subprocess.run(
        [ff, "-y", "-i", inp, "-c:v", "copy", "-bsf:v", "hevc_mp4toannexb",
         "-f", "hevc", src_hevc],
        capture_output=True, timeout=300, **popen_kw)
    if r.returncode != 0:
        logger.warning("DoVi: HEVC 提取失败")
        return None

    # 2. 提取 RPU
    src_rpu = os.path.join(work_dir, "_dovi_src_rpu.bin")
    r = subprocess.run([dovi, "extract-rpu", src_hevc, "-o", src_rpu],
                       capture_output=True, timeout=60, **popen_kw)
    os.unlink(src_hevc)
    if r.returncode != 0:
        logger.warning("DoVi: RPU 提取失败")
        return None

    # 3. 获取实际 RPU 帧数: 用小 duplicate 触发 metadata len 输出
    rpu_frames = 0
    import re as _re
    cnt_json = os.path.join(work_dir, "_dovi_cnt.json")
    cnt_out = os.path.join(work_dir, "_dovi_cnt.bin")
    with open(cnt_json, "w", encoding="utf-8") as f:
        json.dump({"duplicate": [{"source": 0, "offset": 100, "length": 1}]}, f)
    r = subprocess.run(
        [dovi, "editor", "-i", src_rpu, "-j", cnt_json, "-o", cnt_out],
        capture_output=True, timeout=15, **popen_kw)
    m = _re.search(r"Initial metadata len (\d+)",
                   r.stderr.decode(errors="replace"))
    if m:
        rpu_frames = int(m.group(1))
    for f in (cnt_json, cnt_out):
        try:
            os.unlink(f)
        except Exception:
            pass
    if rpu_frames <= 0:
        rpu_frames = total_source_frames
    logger.info("DoVi: RPU 实际 %d 帧 (视频 %d 帧)", rpu_frames, total_source_frames)

    # 4. 生成匹配输出帧数的 RPU (duplicate 源 RPU 序列)
    expanded_rpu = os.path.join(work_dir, "_dovi_expanded.bin")
    edit_json = os.path.join(work_dir, "_dovi_edit.json")
    with open(edit_json, "w", encoding="utf-8") as f:
        json.dump({
            "duplicate": [{
                "source": 0,
                "offset": rpu_frames,
                "length": rpu_frames
            }]
        }, f)

    r = subprocess.run(
        [dovi, "editor", "-i", src_rpu, "-j", edit_json, "-o", expanded_rpu],
        capture_output=True, timeout=60, **popen_kw)
    os.unlink(src_rpu)
    os.unlink(edit_json)
    if r.returncode != 0:
        logger.warning("DoVi: RPU 扩展失败: %s",
                       r.stderr.decode(errors="replace")[-200:])
        return None

    logger.info("DoVi: RPU 处理完成 (%d→%d 条目)",
                total_source_frames, total_output_frames)
    return expanded_rpu


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
            audio_src = inp  # 始终用原始文件提取音轨

            # 保存源属性 (VFR→CFR 后会丢失)
            _has_dovi = reader.has_dovi
            _vfr = reader.vfr_needs_cfr

            # VFR → CFR 预处理 (方案4: 社区公认最可靠方法)
            if _vfr:
                from video_io.reader import convert_vfr_to_cfr
                cfr_path = convert_vfr_to_cfr(inp, reader.fps,
                                              str(out_path.parent))
                reader.close()
                reader = VideoReader(cfr_path)
                reader._cfr_path = cfr_path  # 标记临时文件以便清理

            fps_out = reader.fps * self.fps_multiplier

            # DoVi RPU 预处理
            dovi_rpu = None
            if _has_dovi:
                total_out_est = reader.total_frames * self.fps_multiplier
                dovi_rpu = _process_dovi_rpu(
                    inp, reader.total_frames, total_out_est,
                    str(out_path.parent))
                if dovi_rpu:
                    logger.info("DoVi: RPU 已就绪，将嵌入输出视频")

            # 续传: 读进度文件
            progress_file = out_path.parent / f"{out_path.stem}_progress.txt"
            skip_frames = 0
            if progress_file.exists():
                try:
                    skip_frames = int(progress_file.read_text(encoding="utf-8").strip())
                    logger.info("续传: 跳过%d帧", skip_frames)
                    self.file_resumed.emit(vid, skip_frames)
                except Exception:
                    pass

            total_out = reader.total_frames * self.fps_multiplier

            writer = None
            prev = None
            out_idx = skip_frames
            in_idx = skip_frames // self.fps_multiplier

            # 跳到续传帧
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
                            color_primaries=getattr(reader, 'color_primaries', 'bt709'),
                            dovi_rpu=dovi_rpu)
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
                                     color_primaries=getattr(reader, 'color_primaries', 'bt709'),
                                     dovi_rpu=dovi_rpu)
                writer.write(prev)
            if writer:
                writer.close()
                if self._cancelled:
                    return
            elapsed = time.perf_counter() - t0
            logger.info("完成: %s (%d帧, %.1fs)",
                        Path(inp).name, out_idx, elapsed)
            self.file_finished.emit(vid, out)

            # 清理 VFR→CFR 临时文件 和 DoVi RPU 临时文件
            if getattr(reader, '_cfr_path', None):
                try:
                    os.unlink(reader._cfr_path)
                except Exception:
                    pass
            if dovi_rpu:
                try:
                    os.unlink(dovi_rpu)
                except Exception:
                    pass

        except Exception as e:
            logger.exception("失败: %s", inp)
            self.file_error.emit(vid, str(e))
