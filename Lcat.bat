@echo off
set "APP_DIR=%~dp0"
set "PYTHON=%APP_DIR%python\pythonw.exe"

:: DLL / 可执行文件搜索路径
set "PATH=%APP_DIR%python;%APP_DIR%runtime\torch\lib;%APP_DIR%runtime\av.libs;%APP_DIR%runtime\PyQt5\Qt5\bin;%APP_DIR%runtime\cv2;%APP_DIR%;%PATH%"

:: Python 模块搜索路径
set "PYTHONPATH=%APP_DIR%src;%APP_DIR%runtime"

cd /d "%APP_DIR%"
start "" "%PYTHON%" "%APP_DIR%src\main.py"
