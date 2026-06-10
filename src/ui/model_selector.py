"""模型选择下拉框"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget


class ModelSelector(QWidget):
    model_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        l = QHBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(QLabel("模型:"))
        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self.model_changed.emit)
        l.addWidget(self.combo, 1)

    def populate(self, models):
        self.combo.blockSignals(True)
        self.combo.clear()
        for m in models:
            self.combo.addItem(f"{m['name']} ({m['version'] or '?'})", m)
        self.combo.blockSignals(False)

    def set_current_by_version(self, ver):
        for i in range(self.combo.count()):
            if ver in self.combo.itemText(i):
                self.combo.setCurrentIndex(i)
                return
