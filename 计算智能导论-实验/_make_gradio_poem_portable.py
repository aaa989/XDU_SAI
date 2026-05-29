"""将吟诗作赋Gradio所需文件复制到独立文件夹。"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DST = ROOT / "吟诗作赋_Gradio迁移"
SRC_POEM = ROOT / "吟诗作赋"
REF_POETRY = ROOT / "course_lab_reference-main" / "course_lab_data" / "poem_generation" / "chinese-poetry"

LAB_PATHS_PORTABLE = '''"""课程实验路径（便携包内以本文件所在目录为根）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = ROOT / "course_lab_data"
POETRY_DIR = REF / "poem_generation" / "chinese-poetry"

DIR_POEM = ROOT / "吟诗作赋"
DIR_MUSIC = ROOT / "音乐去噪"
DIR_IMAGE = ROOT / "图像修复"

CKPT_MUSIC = DIR_MUSIC / "checkpoints" / "model.pt"
CKPT_IMAGE = DIR_IMAGE / "checkpoints" / "inpaint_unet.pt"
CKPT_POEM = DIR_POEM / "checkpoints" / "char_lstm.pt"

TEST_IMAGE = REF / "image_restoration" / "test" / "damaged" / "img_inpainting.jpg"
TEST_AUDIO = REF / "music_denoising" / "test" / "raw.MP3"
'''

README = """吟诗作赋 · Gradio 便携包
============================

1) 安装依赖:
   pip install -r requirements.txt

2) 若尚无权重，在「吟诗作赋」目录训练:
   cd 吟诗作赋
   python train.py

3) 启动 Gradio:
   python app_gradio.py
"""


def main() -> None:
    if not SRC_POEM.is_dir():
        print(f"缺少目录: {SRC_POEM}", file=sys.stderr)
        sys.exit(1)

    if not REF_POETRY.is_dir():
        print(f"缺少语料目录: {REF_POETRY}", file=sys.stderr)
        sys.exit(1)

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    for name in ("app_gradio.py", "train_utils.py"):
        src = ROOT / name
        if not src.is_file():
            print(f"缺少文件: {src}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(src, DST / name)

    (DST / "lab_paths.py").write_text(LAB_PATHS_PORTABLE, encoding="utf-8")
    (DST / "README.txt").write_text(README, encoding="utf-8")
    (DST / "requirements.txt").write_text("torch\ngradio\ntqdm\n", encoding="utf-8")

    poem_dst = DST / "吟诗作赋"
    poem_dst.mkdir(parents=True)

    for fn in ("train.py", "infer.py", "model.py", "corpus.py"):
        shutil.copy2(SRC_POEM / fn, poem_dst / fn)

    ck_src = SRC_POEM / "checkpoints"
    ck_dst = poem_dst / "checkpoints"
    if ck_src.is_dir():
        shutil.copytree(ck_src, ck_dst)
    else:
        ck_dst.mkdir(parents=True)
        (ck_dst / ".gitkeep").write_text("", encoding="utf-8")

    poetry_dst = DST / "course_lab_data" / "poem_generation" / "chinese-poetry"
    poetry_dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在复制语料: {REF_POETRY} -> {poetry_dst}")
    shutil.copytree(REF_POETRY, poetry_dst)

    print(f"完成。输出目录: {DST}")


if __name__ == "__main__":
    main()
