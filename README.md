# Lcat — 自用视频补帧工具

基于 [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) PyTorch 架构的视频插帧软件，PyQt5 图形界面。

## 功能

- 2x/4x/8x 视频补帧
- 11 个 RIFE v4.x 模型，放入文件夹即自动识别
- 10 种编码格式 (H.264/H.265/AV1 × 软编/NVENC)
- 音轨/字幕保留，PTS 偏移声画同步
- 批量处理 + 断点续传 + 自动关机
- GPU 自适应推理分辨率
- 拖拽导入、实时进度

## 系统要求

- Windows 10/11 64-bit
- NVIDIA GPU (CUDA 驱动) 或 CPU
- 约 10GB 磁盘空间（含模型）

## 发布页面下载完整包

完整包含 Python 运行时、依赖、模型、ffmpeg，解压即用。
