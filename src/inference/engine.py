"""推理引擎 — 参照 Practical-RIFE 官方实现。

x2/x4/x8 均使用直接 timestep，不做递归二分。
模型 v4.x 的 inference() 接口: inference(img0, img1, timestep, scale)
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_INFER_SIZES = [(1408, 2496), (1088, 1920), (720, 1280)]


class InferenceEngine:
    """RIFE 推理引擎。"""

    def __init__(self, model):
        self.model = model
        self.device = next(model.flownet.parameters()).device
        self._infer_size_idx = 0

    def _probe_size(self, h, w):
        if self.device.type != "cuda":
            return h, w
        for i, (ih, iw) in enumerate(_INFER_SIZES):
            try:
                t0 = torch.randn(1, 3, ih, iw, device=self.device)
                t1 = torch.randn(1, 3, ih, iw, device=self.device)
                _ = self.model.inference(t0, t1, scale=1.0)
                del t0, t1, _
                torch.cuda.empty_cache()
                self._infer_size_idx = i
                return min(ih, h), min(iw, w)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
        return min(720, h), min(1280, w)

    @torch.inference_mode()
    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray,
                    timestep: float = 0.5) -> np.ndarray:
        h, w = frame0.shape[:2]
        max_dim = max(h, w)

        if max_dim <= 1920:
            return self._full_inference(frame0, frame1, timestep, h, w)

        infer_h, infer_w = self._probe_size(h, w)
        scale_used = min(infer_h / h, infer_w / w)
        logger.info("缩放推理: %dx%d -> %dx%d (%.2fx)", w, h, infer_w, infer_h, scale_used)

        t0 = torch.from_numpy(frame0).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        t1 = torch.from_numpy(frame1).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        t0_s = F.interpolate(t0, (infer_h, infer_w), mode="bilinear", align_corners=False)
        t1_s = F.interpolate(t1, (infer_h, infer_w), mode="bilinear", align_corners=False)
        out = self.model.inference(t0_s, t1_s, timestep, scale=1.0)
        out = F.interpolate(out.clamp(0, 1), (h, w), mode="bilinear", align_corners=False)
        return (out[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def _full_inference(self, frame0, frame1, timestep, h, w):
        t0 = torch.from_numpy(frame0).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        t1 = torch.from_numpy(frame1).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        tmp = 128
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        t0 = F.pad(t0, (0, pw - w, 0, ph - h))
        t1 = F.pad(t1, (0, pw - w, 0, ph - h))
        out = self.model.inference(t0, t1, timestep, scale=1.0)
        out = out[:, :, :h, :w].clamp(0, 1)
        return (out[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def interpolate_multi(self, frame0: np.ndarray, frame1: np.ndarray,
                          n: int = 2) -> list:
        """n=2→1帧, n=4→3帧, n=8→7帧。
        参照 Practical-RIFE v4.x: 直接 timestep，不做递归。"""
        timesteps = [i / n for i in range(1, n)]
        return [self.interpolate(frame0, frame1, t) for t in timesteps]
