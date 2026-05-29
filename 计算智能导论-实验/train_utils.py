"""GPU训练公用：设备解析与DataLoader worker数量。"""
from __future__ import annotations

import os
import sys
import torch


def cuda_amp_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def cuda_amp_autocast():
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def resolve_train_device(device: str = "auto") -> torch.device:
    raw = (device or "auto").strip()
    key = raw.lower()
    if key in ("auto", ""):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if key.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到 CUDA：无法按 GPU 训练。请安装 CUDA 版 PyTorch（见 https://pytorch.org/get-started/locally/），"
                "或改用 --device auto / --device cpu。"
            )
        return torch.device(raw)
    return torch.device(raw)


def dataloader_workers() -> int:
    env = os.environ.get("LAB_TRAIN_WORKERS")
    if env is not None:
        try:
            return max(0, int(env.strip()))
        except ValueError:
            pass
    if sys.platform == "win32":
        return 0
    return min(8, max(2, (os.cpu_count() or 8) // 2))


def print_train_device(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"[设备] GPU {name} | 显存约 {total:.1f} GiB")
    else:
        print("[设备] CPU")
