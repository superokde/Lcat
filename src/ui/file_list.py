"""拖拽导入文件列表"""
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QListWidget, QListWidgetItem,
                              QPushButton, QVBoxLayout, QWidget, QHBoxLayout)

SUPPORTED_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


class DropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(self.InternalMove)
        self.setDefaultDropAction(Qt.CopyAction)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            self._add_urls(e.mimeData().urls())
            e.acceptProposedAction()
        else:
            super().dropEvent(e)

    def _add_urls(self, urls):
        for url in urls:
            p = Path(url.toLocalFile())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
                item = QListWidgetItem(p.name)
                item.setData(1, str(p))
                self.addItem(item)
            elif p.is_dir():
                for f in sorted(p.iterdir()):
                    if f.suffix.lower() in SUPPORTED_EXT:
                        item = QListWidgetItem(f.name)
                        item.setData(1, str(f))
                        self.addItem(item)


class FileListWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        l = QVBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        bl = QHBoxLayout()
        self.add_btn = QPushButton("添加视频")
        self.add_btn.clicked.connect(self._add)
        self.remove_btn = QPushButton("移除选中")
        self.remove_btn.clicked.connect(self._remove)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(lambda: self.list.clear())
        bl.addWidget(self.add_btn)
        bl.addWidget(self.remove_btn)
        bl.addWidget(self.clear_btn)
        bl.addStretch()
        l.addLayout(bl)

        self.list = DropListWidget()
        self.list.setSelectionMode(self.list.ExtendedSelection)
        l.addWidget(self.list)

    def _add(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频", "",
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv)")
        for f in files:
            item = QListWidgetItem(Path(f).name)
            item.setData(1, f)
            self.list.addItem(item)

    def _remove(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def get_file_paths(self) -> list[Path]:
        return [Path(self.list.item(i).data(1))
                for i in range(self.list.count())]

    def set_status(self, idx, text):
        if idx < self.list.count():
            self.list.item(idx).setText(
                f"[{text}] {Path(self.list.item(idx).data(1)).name}")

    def get_status(self, idx) -> str:
        if idx < self.list.count():
            t = self.list.item(idx).text()
            if t.startswith("["):
                return t[1:t.index("]")]
        return ""

    def set_status_tooltip(self, idx, tip):
        if idx < self.list.count():
            self.list.item(idx).setToolTip(tip)

    def set_buttons_enabled(self, e):
        for w in [self.add_btn, self.remove_btn, self.clear_btn]:
            w.setEnabled(e)
