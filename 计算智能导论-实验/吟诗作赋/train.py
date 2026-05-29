"""在chinese-poetry语料上训练字符级LSTM。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_utils import (
    cuda_amp_autocast,
    cuda_amp_grad_scaler,
    dataloader_workers,
    print_train_device,
    resolve_train_device,
)

from corpus import build_text_corpus, load_poems
from model import CharLSTM


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int, stoi: Dict[str, int]):
        self.seq_len = seq_len
        self.data = [stoi[c] for c in text if c in stoi]
        self.stoi = stoi

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, i: int):
        chunk = self.data[i : i + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_vocab(text: str) -> tuple[Dict[str, int], List[str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    return stoi, itos


def _infer_meta(stoi: Dict[str, int], itos: List[str], vocab_size: int, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "stoi": stoi,
        "itos": itos,
        "seq_len": args.seq_len,
        "vocab_size": vocab_size,
        "hidden": args.hidden,
        "num_layers": args.layers,
        "emb_dim": args.emb_dim,
        "max_json_files": args.max_json_files,
    }


def _save_best(path: Path, model: nn.Module, meta: Dict[str, Any]) -> None:
    torch.save({"model": model.state_dict(), **meta}, path)


def _save_last(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    resume_next_epoch: int,
    best_loss: float,
    meta: Dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": resume_next_epoch,
        "best_loss": best_loss,
        "scaler": scaler.state_dict() if scaler is not None else None,
        **meta,
    }
    torch.save(payload, path)


def _load_resume(
    path: Path,
    model: CharLSTM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> tuple[int, float, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    resume_next = int(ckpt.get("epoch", 0))
    best = float(ckpt.get("best_loss", float("inf")))
    scaler = cuda_amp_grad_scaler() if use_amp else None
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return resume_next, best, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--emb_dim", type=int, default=192)
    ap.add_argument("--max_json_files", type=int, default=120)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", nargs="?", const="last", default=None)
    args = ap.parse_args()

    device = resolve_train_device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    print_train_device(device)

    here = Path(__file__).resolve().parent
    ckpt_dir = here / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = ckpt_dir / "char_lstm.pt"
    ckpt_last = ckpt_dir / "char_lstm_last.pt"

    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = ckpt_last if args.resume == "last" else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"找不到续训文件: {resume_path}")

    from lab_paths import POETRY_DIR
    lines = load_poems(POETRY_DIR, max_files=args.max_json_files)
    if not lines:
        raise RuntimeError(f"未读取到诗词文本，请检查路径: {POETRY_DIR}")
    text = build_text_corpus(lines)
    stoi, itos = build_vocab(text)
    vocab_size = len(itos)
    print(f"字典大小={vocab_size}, 总字数={len(text)}, 诗句数={len(lines)}")

    ds = CharDataset(text, args.seq_len, stoi)
    nw = dataloader_workers()
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
    )

    model = CharLSTM(
        vocab_size, emb_dim=args.emb_dim, hidden=args.hidden, num_layers=args.layers
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler: Any = cuda_amp_grad_scaler() if use_amp else None

    start_epoch = 0
    best = float("inf")
    meta = _infer_meta(stoi, itos, vocab_size, args)

    if resume_path is not None:
        start_epoch, best, scaler = _load_resume(resume_path, model, opt, device, use_amp)
        print(f"[续训] 从第 {start_epoch + 1} 轮开始，最佳损失={best:.4f}")

    for epoch in range(start_epoch, args.epochs):
        try:
            model.train()
            total_loss, n_samples = 0.0, 0

            for x, y in tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                if use_amp and scaler is not None:
                    with cuda_amp_autocast():
                        logits, _ = model(x)
                        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits, _ = model(x)
                    loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)

            avg_loss = total_loss / max(1, n_samples)
            print(f"epoch {epoch+1} 损失: {avg_loss:.4f}")

        except KeyboardInterrupt:
            print("\n[中断] 正在保存断点…")
            _save_last(ckpt_last, model, opt, scaler, epoch, best, meta)
            raise SystemExit(130) from None

        if avg_loss < best:
            best = avg_loss
            _save_best(ckpt_best, model, meta)
            print(f"✅ 保存最佳模型: {ckpt_best}")

        _save_last(ckpt_last, model, opt, scaler, epoch + 1, best, meta)

    print("🎉 训练结束！最佳权重:", ckpt_best)


if __name__ == "__main__":
    main()
