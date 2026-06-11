"""模型扫描 + 加载 (使用模型自带 RIFE_HDv3.py 架构)"""
import logging
import sys
import gc
from pathlib import Path
import torch

from utils.config import MODEL_DIR_OFFICIAL, MODEL_DIR_CUSTOM, get_model_version

logger = logging.getLogger(__name__)


def _find_flownet_pkl(model_dir: Path) -> Path | None:
    d = model_dir / "train_log" / "flownet.pkl"
    if d.exists():
        return d
    for p in sorted(model_dir.glob("*/train_log/flownet.pkl")):
        return p
    return None


def _cleanup_model_paths():
    """清理之前加载的模型在 sys.path 和 sys.modules 中的残留。"""
    import importlib
    for p in list(sys.path):
        if "rife_official" in str(p) or "rife_custom" in str(p):
            sys.path.remove(p)
    for m in list(sys.modules):
        if m.startswith("train_log") or m.startswith("rife_model_"):
            del sys.modules[m]
    # 清除 importlib 缓存，确保下次导入重新加载
    importlib.invalidate_caches()


def _is_prelu_model(pkl_path: Path) -> bool:
    try:
        sd = torch.load(str(pkl_path), map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        return "block0.conv0.0.1.weight" in sd
    except Exception:
        return False


# v4.20 的 IFNet_HDv3.py 内部有自引用 teacher IFNet 构造参数不匹配的 bug
_SKIP_VERSIONS = {"v4.20"}


def scan_models() -> list[dict]:
    models = []
    seen = set()
    for base, src in [(MODEL_DIR_OFFICIAL, "official"), (MODEL_DIR_CUSTOM, "custom")]:
        if not base.exists():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            pkl = _find_flownet_pkl(entry)
            if pkl is None:
                continue
            if _is_prelu_model(pkl):
                logger.info("跳过 PReLU: %s", entry.name)
                continue
            ver = get_model_version(entry.name)
            if ver in _SKIP_VERSIONS:
                logger.info("跳过不兼容版本: %s", entry.name)
                continue
            ver = get_model_version(entry.name)
            key = ver or entry.name
            if key in seen:
                continue
            seen.add(key)
            models.append({"name": entry.name, "model_dir": entry,
                           "pkl_path": pkl, "version": ver, "source": src})
    return models


def load_model(model_info: dict, device: torch.device):
    model_dir = model_info["model_dir"]
    train_log = model_info["pkl_path"].parent
    arch_file = train_log / "RIFE_HDv3.py"
    if not arch_file.exists():
        raise FileNotFoundError(f"模型架构文件缺失: {arch_file}")

    # 确保 model_dir 和项目根在 sys.path 中
    # 清理旧模型的 sys.path 和 sys.modules 缓存，防止不同模型架构冲突
    _cleanup_model_paths()
    # 模型目录 + train_log 父目录都加入 sys.path，兼容嵌套结构
    sys.path.insert(0, str(model_dir))
    sys.path.insert(0, str(train_log.parent))

    from utils.config import APP_ROOT
    if str(APP_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(APP_ROOT / "src"))

    # 动态导入模型自带的 RIFE_HDv3.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"rife_model_{model_info['name']}", arch_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    model = mod.Model()
    if not hasattr(model, "version"):
        model.version = 0
    model.load_model(str(train_log), -1)
    model.eval()
    model.device()
    logger.info("已加载: %s (v%s, %d params)", model_info["name"],
                model_info["version"] or "?",
                sum(p.numel() for p in model.flownet.parameters()))
    return model


def unload_model(model):
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
