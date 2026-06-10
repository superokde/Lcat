"""Worker: 后台推理线程 (单管道直写 + 断点续传)"""
import logging
import time
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal
from video_io.reader import VideoReader
from video_io.writer import VideoWriter

logger = logging.getLogger(__name__)


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
            fps_out = reader.fps * self.fps_multiplier

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
                            out, fps_out, w, h, inp,
                            self.encoder, self.crf, self.pix_fmt,
                            skip_frames=skip_frames)
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
                writer = VideoWriter(out, fps_out, w, h, inp,
                                     self.encoder, self.crf, self.pix_fmt)
                writer.write(prev)
            if writer:
                writer.close()
                if self._cancelled:
                    return
            elapsed = time.perf_counter() - t0
            logger.info("完成: %s (%d帧, %.1fs)",
                        Path(inp).name, out_idx, elapsed)
            self.file_finished.emit(vid, out)

        except Exception as e:
            logger.exception("失败: %s", inp)
            self.file_error.emit(vid, str(e))
