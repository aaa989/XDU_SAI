"""课程实验路径配置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = ROOT / "course_lab_reference-main" / "course_lab_data"
TEST_IMAGE = REF / "image_restoration" / "test" / "damaged" / "img_inpainting.jpg"
TEST_AUDIO = REF / "music_denoising" / "test" / "raw.MP3"
POETRY_DIR = REF / "poem_generation" / "chinese-poetry"

DIR_MUSIC = ROOT / "音乐去噪"
DIR_IMAGE = ROOT / "图像修复"
DIR_POEM = ROOT / "吟诗作赋"

CKPT_MUSIC = DIR_MUSIC / "checkpoints" / "model.pt"
CKPT_IMAGE = DIR_IMAGE / "checkpoints" / "inpaint_unet.pt"
CKPT_POEM = DIR_POEM / "checkpoints" / "char_lstm.pt"
