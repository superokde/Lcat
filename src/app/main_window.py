"""主窗口"""
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QHBoxLayout, QMainWindow, QMessageBox,
                              QProgressBar, QVBoxLayout, QWidget)

from app.worker import InterpolationWorker
from inference.engine import InferenceEngine
from inference.model_loader import scan_models, load_model, unload_model
from ui.controls_panel import ControlsPanel
from ui.file_list import FileListWidget
from ui.model_selector import ModelSelector

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自用视频插帧工具")
        self.setMinimumSize(1024, 640)
        self._set_app_icon()
        self._model = None
        self._engine = None
        self._thread = None
        self._worker = None
        self._models = []
        self._post_action = "无"
        self._file_start_time = 0.0
        self._setup_ui()
        self._refresh_models()

    @staticmethod
    def _icon_path() -> Path:
        from utils.config import APP_ROOT
        return APP_ROOT / "icon" / "cat.png"

    def _set_app_icon(self):
        p = self._icon_path()
        if p.exists():
            icon = QIcon(str(p))
            self.setWindowIcon(icon)
            from PyQt5.QtWidgets import QApplication
            QApplication.instance().setWindowIcon(icon)

    def _setup_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)

        self.model_selector = ModelSelector()
        self.model_selector.model_changed.connect(self._on_model_changed)
        root.addWidget(self.model_selector)

        body = QHBoxLayout()
        self.file_list = FileListWidget()
        body.addWidget(self.file_list, 2)
        self.controls = ControlsPanel()
        self.controls.start_btn.clicked.connect(self._on_start)
        self.controls.stop_btn.clicked.connect(self._on_stop)
        self.controls.device_combo.currentIndexChanged.connect(
            lambda: self._load_current_model())
        body.addWidget(self.controls, 1)
        root.addLayout(body)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

    def _refresh_models(self):
        self._models = scan_models()
        self.model_selector.combo.blockSignals(True)
        self.model_selector.populate(self._models)
        self.model_selector.set_current_by_version("v4.25")
        if self.model_selector.combo.currentIndex() < 0 and self._models:
            self.model_selector.combo.setCurrentIndex(0)
        self.model_selector.combo.blockSignals(False)
        if self.model_selector.combo.currentIndex() >= 0:
            self._load_current_model()

    def _on_model_changed(self, idx):
        if idx >= 0:
            self._load_current_model()

    def _load_current_model(self):
        idx = self.model_selector.combo.currentIndex()
        if idx < 0:
            return
        info = self._models[idx]
        logger.info("加载: %s", info["name"])
        self._engine = None
        unload_model(self._model)
        self._model = None
        try:
            import torch
            dev_str = self.controls.selected_device()
            has_cuda = torch.cuda.is_available()
            device = torch.device(dev_str) if has_cuda else torch.device("cpu")
            self._model = load_model(info, device)
            if self._model is None:
                QMessageBox.warning(self, "失败", f"无法加载: {info['name']}")
                return
            p = self.controls.get_params()
            self._engine = InferenceEngine(self._model,
                scene_threshold=p.get("scene_threshold", 0.2),
                use_fp16=p.get("use_fp16", False))
            dn = (torch.cuda.get_device_name(device.index)
                  if device.type == "cuda" else "CPU")
            self.setWindowTitle(f"自用视频插帧工具 - {info['name']} [{dn}]")
            logger.info("就绪: %s on %s", info["name"], dn)
        except Exception as e:
            logger.exception("加载崩溃")
            QMessageBox.critical(self, "崩溃", str(e))

    def _gen_out(self, inp, od):
        base = od / f"{inp.stem}_interpolated{inp.suffix}"
        if not base.exists():
            return base
        c = 1
        while True:
            cand = od / f"{inp.stem}_interpolated_{c}{inp.suffix}"
            if not cand.exists():
                return cand
            c += 1

    def _on_start(self):
        if self._engine is None:
            QMessageBox.warning(self, "提示", "请先选择模型")
            return
        fps = self.file_list.get_file_paths()
        if not fps:
            QMessageBox.warning(self, "提示", "请添加视频")
            return
        p = self.controls.get_params()
        od = p.get("output_dir", "")
        if not od:
            QMessageBox.warning(self, "提示", "请选择输出目录")
            return

        self._post_action = p.get("post_action", "无")
        out_dir = Path(od)
        out_dir.mkdir(parents=True, exist_ok=True)
        fld = [(i, str(x), str(self._gen_out(x, out_dir)))
               for i, x in enumerate(fps)]
        for i in range(len(fps)):
            self.file_list.set_status(i, "等待中")

        self._thread = QThread()
        self._worker = InterpolationWorker(
            fld, self._engine, p["fps_multiplier"],
            p.get("encoder", "libx264"), p.get("crf", 18),
            p.get("pix_fmt", "yuv420p"))
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.file_error.connect(self._on_file_error)
        self._worker.file_resumed.connect(self._on_file_resumed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.all_finished.connect(self._on_all_finished)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.controls.start_btn.setEnabled(False)
        self.controls.stop_btn.setEnabled(True)
        self.controls.set_controls_enabled(False)
        self.model_selector.combo.setEnabled(False)
        self.file_list.set_buttons_enabled(False)
        self._thread.start()

    def _on_stop(self):
        if self._worker:
            self._worker.cancel()
            self.controls.stop_btn.setEnabled(False)
            self.controls.stop_btn.setText("停止中...")

    def _on_progress(self, c, t):
        self.progress_bar.setMaximum(t)
        self.progress_bar.setValue(c)

    def _on_file_started(self, vid):
        self.file_list.set_status(vid, "处理中")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备中...")
        self._file_start_time = time.perf_counter()

    def _on_file_progress(self, vid, c, t):
        self.file_list.set_status(vid, f"处理中")
        if c > 0 and t > 0:
            e = time.perf_counter() - self._file_start_time
            if e > 1:
                s = c / e
                r = (t - c) / s
                if r < 60:
                    eta = f"剩余 {int(r)}s"
                elif r < 3600:
                    eta = f"剩余 {int(r // 60)}m{int(r % 60)}s"
                else:
                    eta = f"剩余 {int(r // 3600)}h{int((r % 3600) // 60)}m"
                self.progress_bar.setFormat(
                    f"{c}/{t} | {eta} | {s:.1f}fps")

    def _on_file_finished(self, vid, out):
        self.file_list.set_status(vid, "完成")

    def _on_file_error(self, vid, msg):
        self.file_list.set_status(vid, "失败")
        self.file_list.set_status_tooltip(vid, msg)

    def _on_file_resumed(self, vid, skipped):
        self.file_list.set_status(vid, f"续传(跳过{skipped}帧)")

    def _on_cancelled(self):
        QMessageBox.information(self, "已停止", "任务已取消。")

    def _on_all_finished(self):
        try:
            self.progress_bar.setVisible(False)
            self.controls.start_btn.setEnabled(True)
            self.controls.stop_btn.setEnabled(False)
            self.controls.stop_btn.setText("强制停止")
            self.controls.set_controls_enabled(True)
            self.model_selector.combo.setEnabled(True)
            self.file_list.set_buttons_enabled(True)
            self.setWindowTitle("自用视频插帧工具")

            fc = len(self.file_list.get_file_paths())
            dn = sum(1 for i in range(fc)
                     if self.file_list.get_status(i) == "完成")
            fl = sum(1 for i in range(fc)
                     if self.file_list.get_status(i) == "失败")

            if self._post_action == "关机":
                if fl > 0:
                    QMessageBox.warning(
                        self, "有失败任务, 取消关机",
                        f"{dn}成功 {fl}失败 共{fc}\n\n因存在失败任务，已取消自动关机。")
                else:
                    self._auto_shutdown()
            else:
                msg = (f"全部成功: {dn}/{fc}" if fl == 0
                       else f"{dn}成功 {fl}失败 共{fc}")
                QMessageBox.information(self, "完成", msg)
        except Exception:
            logger.exception("_on_all_finished 崩溃")
        finally:
            if self._thread and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(5000)
            if self._worker:
                self._worker.deleteLater()
            self._worker = None
            self._thread = None

    def _auto_shutdown(self):
        try:
            subprocess.run(["shutdown", "/s", "/t", "60",
                            "/c", "补帧完成"],
                           check=True, capture_output=True, text=True,
                           timeout=30)
        except Exception:
            logger.exception("关机命令失败")
            return
        reply = QMessageBox.warning(
            self, "补帧完成",
            "任务已完成，60秒后自动关机。\n\n点击「取消」中止关机。",
            QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            subprocess.run(["shutdown", "/a"], capture_output=True, timeout=10)

    def closeEvent(self, e):
        try:
            if self._worker:
                self._worker.cancel()
                self._worker.deleteLater()
            unload_model(self._model)
            self._model = None
        except Exception:
            logger.exception("closeEvent 崩溃")
        super().closeEvent(e)
