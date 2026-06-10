"""入口"""
import logging
import sys
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow


def _set_app_id():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "lcat.frame.interpolator")
        except Exception:
            pass


def main():
    _set_app_id()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("app.log", encoding="utf-8")])

    logger = logging.getLogger(__name__)
    logger.info("启动应用...")

    try:
        import torch
        logger.info("PyTorch %s, CUDA %s, GPU: %s",
                     torch.__version__,
                     torch.version.cuda if torch.cuda.is_available() else "N/A",
                     torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "none")
    except Exception as e:
        logger.warning("PyTorch启动失败 (仅CPU可用): %s", e)

    app = QApplication(sys.argv)
    app.setApplicationName("自用视频插帧工具")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
