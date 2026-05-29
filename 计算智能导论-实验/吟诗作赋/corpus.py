"""从chinese-poetry语料构建训练文本。"""
from __future__ import annotations

import json
import random
from pathlib import Path


def load_poems(poetry_root: Path, max_files: int = 80) -> list[str]:
    json_files = sorted(poetry_root.rglob("*.json"))
    random.seed(42)
    random.shuffle(json_files)
    json_files = json_files[:max_files]

    lines: list[str] = []

    for fp in json_files:
        try:
            raw = fp.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "paragraphs" in item:
                    for p in item["paragraphs"]:
                        p = str(p).strip()
                        if len(p) >= 4:
                            lines.append(p)
        elif isinstance(data, dict) and "paragraphs" in data:
            for p in data["paragraphs"]:
                p = str(p).strip()
                if len(p) >= 4:
                    lines.append(p)

    return lines


def build_text_corpus(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"
