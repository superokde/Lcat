"""控制面板: 输出目录/参数/编码/设备/操作"""
import torch
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox,
                              QHBoxLayout, QLineEdit, QPushButton, QSpinBox,
                              QVBoxLayout, QWidget)

ALL_FORMATS = [
    ("H.264 (软编)", "libx264", "yuv420p"),
    ("H.265 8bit (软编)", "libx265", "yuv420p"),
    ("H.265 10bit (软编)", "libx265", "yuv420p10le"),
    ("AV1 8bit (软编)", "libsvtav1", "yuv420p"),
    ("AV1 10bit (软编)", "libsvtav1", "yuv420p10le"),
    ("H.264 NVENC (硬编)", "h264_nvenc", "yuv420p"),
    ("HEVC NVENC (硬编)", "hevc_nvenc", "yuv420p"),
    ("HEVC NVENC 10bit (硬编)", "hevc_nvenc", "p010le"),
    ("AV1 NVENC (硬编)", "av1_nvenc", "yuv420p"),
    ("AV1 NVENC 10bit (硬编)", "av1_nvenc", "p010le"),
]


def _has_gpu():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


class ControlsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        l = QVBoxLayout(self)

        og = QGroupBox("输出目录")
        ol = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("选择输出目录...")
        self.output_browse = QPushButton("浏览")
        self.output_browse.clicked.connect(lambda: self._browse())
        ol.addWidget(self.output_dir)
        ol.addWidget(self.output_browse)
        og.setLayout(ol)
        l.addWidget(og)

        pg = QGroupBox("参数")
        pf = QFormLayout()
        self.fps_mult = QComboBox()
        self.fps_mult.addItems(["2x", "4x", "8x"])
        pf.addRow("倍数:", self.fps_mult)
        self.crf = QSpinBox()
        self.crf.setRange(0, 51)
        self.crf.setValue(23)
        pf.addRow("CQ:", self.crf)
        self.crf.setToolTip("0=近无损/51=最差, 推荐 21-26")
        self.post_action = QComboBox()
        self.post_action.addItems(["无", "关机"])
        pf.addRow("任务后:", self.post_action)
        # 场景检测阈值
        self.scene_threshold = QSpinBox()
        self.scene_threshold.setRange(1, 100)
        self.scene_threshold.setValue(20)
        self.scene_threshold.setSuffix("%")
        self.scene_threshold.setToolTip("SSIM 阈值: 越低越容易判定为场景切换\n动漫推荐 5%, 真人推荐 20%")
        pf.addRow("场景切换:", self.scene_threshold)
        # 动漫模式
        self.anime_mode = QComboBox()
        self.anime_mode.addItems(["标准", "动漫"])
        self.anime_mode.setToolTip("动漫模式: 场景阈值 5%，减少跨切换 morph 伪影")
        self.anime_mode.currentIndexChanged.connect(self._on_anime_mode)
        pf.addRow("内容类型:", self.anime_mode)
        pg.setLayout(pf)
        l.addWidget(pg)

        eg = QGroupBox("编码")
        ef = QFormLayout()
        self.format_combo = QComboBox()
        for label, enc, pf in ALL_FORMATS:
            self.format_combo.addItem(label, {"encoder": enc, "pix_fmt": pf})
        ef.addRow("格式:", self.format_combo)
        eg.setLayout(ef)
        l.addWidget(eg)

        dg = QGroupBox("设备")
        df = QFormLayout()
        self.device_combo = QComboBox()
        if _has_gpu():
            for i in range(torch.cuda.device_count()):
                self.device_combo.addItem(
                    f"GPU {i}: {torch.cuda.get_device_name(i)}", f"cuda:{i}")
        self.device_combo.addItem("CPU", "cpu")
        df.addRow("推理:", self.device_combo)
        self.fp16_checkbox = QComboBox()
        self.fp16_checkbox.addItems(["FP32 (标准)", "FP16 (高速)"])
        self.fp16_checkbox.setToolTip("FP16 半精度推理，RTX 20 系以上支持\n速度提升约 30%，画质几乎无损")
        df.addRow("精度:", self.fp16_checkbox)
        dg.setLayout(df)
        l.addWidget(dg)

        ag = QGroupBox("操作")
        al = QVBoxLayout()
        self.start_btn = QPushButton("开始批处理")
        self.start_btn.setMinimumHeight(36)
        al.addWidget(self.start_btn)
        self.stop_btn = QPushButton("强制停止")
        self.stop_btn.setMinimumHeight(32)
        self.stop_btn.setStyleSheet("color:red")
        self.stop_btn.setEnabled(False)
        al.addWidget(self.stop_btn)
        ag.setLayout(al)
        l.addWidget(ag)
        l.addStretch()

    def _on_anime_mode(self):
        idx = self.anime_mode.currentIndex()
        if idx == 1:  # 动漫
            self.scene_threshold.setValue(5)
        else:
            self.scene_threshold.setValue(20)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.output_dir.setText(d)

    def get_params(self):
        fmt = self.format_combo.currentData() or {}
        return {
            "output_dir": self.output_dir.text(),
            "fps_multiplier": int(self.fps_mult.currentText().rstrip("x")),
            "crf": self.crf.value(),
            "post_action": self.post_action.currentText(),
            "encoder": fmt.get("encoder", "libx264"),
            "pix_fmt": fmt.get("pix_fmt", "yuv420p"),
            "scene_threshold": self.scene_threshold.value() / 100.0,
            "use_fp16": self.fp16_checkbox.currentIndex() == 1,
        }

    def selected_device(self):
        d = self.device_combo.currentData()
        return d if d else "cuda:0" if _has_gpu() else "cpu"

    def set_controls_enabled(self, e):
        for w in [self.output_dir, self.output_browse, self.fps_mult,
                   self.crf, self.post_action, self.format_combo,
                   self.device_combo, self.scene_threshold, self.anime_mode,
                   self.fp16_checkbox]:
            w.setEnabled(e)
