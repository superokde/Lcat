"""微型启动器 — 参照 PyInstaller 官方最佳实践。

使用 sys._MEIPASS 定位资源文件（适配 --onefile 临时解压目录）。
"""
import os
import subprocess
import sys
import traceback


def resource_path(relative_path: str) -> str:
    """获取资源绝对路径，开发期和 PyInstaller --onefile 均适用。"""
    try:
        base_path = sys._MEIPASS  # PyInstaller 临时解压目录
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def log_error(msg: str):
    """写错误到桌面，方便排查。"""
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        with open(os.path.join(desktop, "lcat_error.log"), "a",
                  encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass


def main():
    # APP_DIR: exe 文件所在目录（非临时解压目录）
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))

    python_exe = os.path.join(APP_DIR, "python", "pythonw.exe")
    main_py = os.path.join(APP_DIR, "src", "main.py")

    if not os.path.isfile(python_exe):
        log_error(f"pythonw.exe 未找到: {python_exe}")
        return
    if not os.path.isfile(main_py):
        log_error(f"main.py 未找到: {main_py}")
        return

    dll_paths = [
        os.path.join(APP_DIR, "python"),
        os.path.join(APP_DIR, "runtime", "torch", "lib"),
        os.path.join(APP_DIR, "runtime", "av.libs"),
        os.path.join(APP_DIR, "runtime", "PyQt5", "Qt5", "bin"),
        os.path.join(APP_DIR, "runtime", "cv2"),
        APP_DIR,
    ]
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(dll_paths) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.join(APP_DIR, "src"),
        os.path.join(APP_DIR, "runtime"),
    ])

    try:
        subprocess.Popen(
            [python_exe, main_py],
            env=env, cwd=APP_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW
            if sys.platform == "win32" else 0,
        )
    except Exception:
        log_error(traceback.format_exc())


if __name__ == "__main__":
    main()
