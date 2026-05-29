"""根据提示续写诗词，支持七言/五言约束。"""
from __future__ import annotations

from pathlib import Path
import torch
from model import CharLSTM

_CLAUSE_DELIMS: frozenset[str] = frozenset("，。、；：！？…\n\r")


def _clause_content_len(text: str) -> int:
    last = -1
    for j in range(len(text) - 1, -1, -1):
        if text[j] in _CLAUSE_DELIMS:
            last = j
            break
    tail = text[last + 1 :]
    return sum(1 for c in tail if c not in _CLAUSE_DELIMS)


def _vocab_delim_masks(itos: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    is_delim = [c in _CLAUSE_DELIMS for c in itos]
    d = torch.tensor(is_delim, dtype=torch.bool, device=device)
    return d, ~d


def _mask_logits_clause(
    logits_1d: torch.Tensor,
    *,
    chars_per_clause: int | None,
    text_so_far: str,
    delim_b: torch.Tensor,
) -> torch.Tensor:
    if chars_per_clause is None:
        return logits_1d

    out = logits_1d.clone()
    L = _clause_content_len(text_so_far)

    if L >= chars_per_clause:
        out = out.masked_fill(~delim_b, float("-inf"))
    else:
        out = out.masked_fill(delim_b, float("-inf"))

    if not torch.isfinite(out).any() or out.max() == float("-inf"):
        return logits_1d
    return out


def _load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.inference_mode()
def generate_lines(
    prompt: str,
    ckpt_path: Path | None = None,
    max_new_chars: int = 120,
    temperature: float = 0.85,
    device: str | None = None,
    chars_per_clause: int | None = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    here = Path(__file__).resolve().parent

    ckpt_path = ckpt_path or (here / "checkpoints" / "char_lstm.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"缺少权重: {ckpt_path}，请先运行 train.py")

    ckpt = _load(ckpt_path, device)
    stoi: dict = ckpt["stoi"]
    itos: list = ckpt["itos"]
    vocab_size = int(ckpt["vocab_size"])
    hidden = int(ckpt.get("hidden", 384))
    n_layers = int(ckpt.get("num_layers", 2))
    emb_dim = int(ckpt.get("emb_dim", 128))

    model = CharLSTM(vocab_size, emb_dim=emb_dim, hidden=hidden, num_layers=n_layers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dev = torch.device(device)
    delim_b, _ = _vocab_delim_masks(itos, dev)
    if chars_per_clause is not None and not delim_b.any():
        chars_per_clause = None

    text = "".join(ch for ch in prompt if ch in stoi)
    if not text:
        text = itos[0] if itos else "春"

    indices = [stoi[c] for c in text]
    inp = torch.tensor([indices], dtype=torch.long, device=device)

    logits, hidden_state = model(inp, None)
    logits = logits[:, -1, :] / max(temperature, 1e-6)

    logits_flat = _mask_logits_clause(
        logits[0],
        chars_per_clause=chars_per_clause,
        text_so_far=text,
        delim_b=delim_b,
    ).unsqueeze(0)

    out_chars = list(text)
    next_i = torch.multinomial(torch.softmax(logits_flat, dim=-1), 1).item()

    n_gen = max(0, min(int(max_new_chars), 65536))
    for _ in range(n_gen):
        next_ch = itos[next_i]
        out_chars.append(next_ch)
        cur = "".join(out_chars)

        inp_t = torch.tensor([[next_i]], dtype=torch.long, device=device)
        logits, hidden_state = model(inp_t, hidden_state)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        logits_flat = _mask_logits_clause(
            logits[0],
            chars_per_clause=chars_per_clause,
            text_so_far=cur,
            delim_b=delim_b,
        ).unsqueeze(0)

        next_i = torch.multinomial(torch.softmax(logits_flat, dim=-1), 1).item()

    return "".join(out_chars)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="春风吹")
    ap.add_argument("--chars", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--clause", type=int, default=None)
    args = ap.parse_args()
    print(generate_lines(args.prompt, max_new_chars=args.chars, temperature=args.temp, chars_per_clause=args.clause))
