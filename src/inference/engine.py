"""推理引擎 — 参照 Practical-RIFE 官方实现。

x2/x4/x8 均使用直接 timestep，不做递归二分。
支持 FP16 推理、可调场景检测阈值、动漫模式。
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_INFER_SIZES = [(1408, 2496), (1088, 1920), (720, 1280)]

# 场景切换检测阈值: SSIM 低于此值视为场景切换
_SCENE_THRESHOLD_DEFAULT = 0.2     # 默认 (适合真人/混合内容)
_SCENE_THRESHOLD_ANIME = 0.05      # 动漫模式 (动漫场景切换更频繁)


def _ssim_matlab(img0, img1):
    """简化 SSIM 计算 (参考 Practical-RIFE)。img0, img1: NCHW float32 [0,1]"""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu0 = F.avg_pool2d(img0, 3, 1)
    mu1 = F.avg_pool2d(img1, 3, 1)
    mu0_sq, mu1_sq = mu0 ** 2, mu1 ** 2
    mu01 = mu0 * mu1
    sigma0 = F.avg_pool2d(img0 ** 2, 3, 1) - mu0_sq
    sigma1 = F.avg_pool2d(img1 ** 2, 3, 1) - mu1_sq
    sigma01 = F.avg_pool2d(img0 * img1, 3, 1) - mu01
    ssim_n = (2 * mu01 + c1) * (2 * sigma01 + c2)
    ssim_d = (mu0_sq + mu1_sq + c1) * (sigma0 + sigma1 + c2)
    return (ssim_n / ssim_d).mean().item()


class InferenceEngine:
    """RIFE 推理引擎。

    Args:
        model: RIFE 模型实例
        scene_threshold: 场景切换 SSIM 阈值 (0~1, 越小越敏感)
        use_fp16: 启用 FP16 推理 (RTX 20 系以上支持，提速约 30%)
    """

    def __init__(self, model, scene_threshold: float = _SCENE_THRESHOLD_DEFAULT,
                 use_fp16: bool = False):
        self.model = model
        self.device = next(model.flownet.parameters()).device
        self._infer_size_idx = 0
        self.scene_threshold = scene_threshold
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        if self.use_fp16:
            logger.info("FP16 推理已启用")

    def _to_tensor(self, frame: np.ndarray):
        t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.use_fp16:
            t = t.half()
        else:
            t = t / 255.0
        return t

    def _from_tensor(self, t: torch.Tensor) -> np.ndarray:
        if self.use_fp16:
            t = t.float()
        t = t.clamp(0, 1 if not self.use_fp16 else 255)
        if self.use_fp16:
            t = t / 255.0  # half 推理结果范围 [0, 255]，归一化到 [0, 1]
        return (t[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

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

        t0 = self._to_tensor(frame0)
        t1 = self._to_tensor(frame1)
        t0_s = F.interpolate(t0.float() if self.use_fp16 else t0,
                             (infer_h, infer_w), mode="bilinear", align_corners=False)
        t1_s = F.interpolate(t1.float() if self.use_fp16 else t1,
                             (infer_h, infer_w), mode="bilinear", align_corners=False)
        out = self.model.inference(t0_s, t1_s, timestep, scale=1.0)
        out = F.interpolate(out.clamp(0, 1), (h, w), mode="bilinear", align_corners=False)
        return (out[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def _full_inference(self, frame0, frame1, timestep, h, w):
        t0 = self._to_tensor(frame0)
        t1 = self._to_tensor(frame1)
        tmp = 128
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        t0 = F.pad(t0.float() if self.use_fp16 else t0, (0, pw - w, 0, ph - h))
        t1 = F.pad(t1.float() if self.use_fp16 else t1, (0, pw - w, 0, ph - h))
        out = self.model.inference(t0, t1, timestep, scale=1.0)
        out = out[:, :, :h, :w].clamp(0, 1)
        return (out[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def interpolate_multi(self, frame0: np.ndarray, frame1: np.ndarray,
                          n: int = 2) -> list:
        """n=2→1帧, n=4→3帧, n=8→7帧。

        场景检测: SSIM 低于阈值时返回重复帧，避免跨场景切换的 morph 伪影。
        参照 Practical-RIFE 官方 scene change detection。
        """
        # SSIM 场景检测 (32×32 降采样快速判断)
        h, w = frame0.shape[:2]
        t0 = torch.from_numpy(frame0).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        t1 = torch.from_numpy(frame1).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        i0s = F.interpolate(t0[:, :3], (32, 32), mode="bilinear", align_corners=False)
        i1s = F.interpolate(t1[:, :3], (32, 32), mode="bilinear", align_corners=False)
        ssim = _ssim_matlab(i0s, i1s)
        del t0, t1

        if ssim < self.scene_threshold:
            return [frame0.copy() for _ in range(n - 1)]

        timesteps = [i / n for i in range(1, n)]
        return [self.interpolate(frame0, frame1, t) for t in timesteps]
