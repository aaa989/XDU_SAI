"""Gradio 演示：字符级 LSTM 古诗续写。"""
from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse


def _ensure_localhost_bypass_proxy() -> None:
    extra = "127.0.0.1,localhost,::1"
    for key in ("NO_PROXY", "no_proxy"):
        cur = (os.environ.get(key) or "").strip()
        if not cur:
            os.environ[key] = extra
            continue
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        for item in extra.split(","):
            if item not in parts:
                parts.append(item)
        os.environ[key] = ",".join(parts)


_ensure_localhost_bypass_proxy()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from lab_paths import CKPT_POEM

GENRE_CHARS_PER_CLAUSE: dict[str, int | None] = {
    "不限": None,
    "七言": 7,
    "五言": 5,
    "词": None,
    "乐府": None,
}
GENRE_INFO_LABEL: dict[str, str] = {
    "不限": "不限",
    "七言": "七言（每句 7 字）",
    "五言": "五言（每句 5 字）",
    "词": "词",
    "乐府": "乐府",
}

EXAMPLES_BY_GENRE: dict[str, list[str]] = {
    "不限": ["春眠不觉晓，", "床前明月光，"],
    "七言": ["春江潮水连海平，", "孤山寺北贾亭西，"],
    "五言": ["锄禾日当午，", "白日依山尽，"],
    "词": ["明月几时有，", "大江东去，"],
    "乐府": ["青青园中葵，", "孔雀东南飞，"],
}
DEFAULT_PROMPT = EXAMPLES_BY_GENRE["不限"][0]

POEM_UI_CSS = """
@media (max-width: 768px) {
  .gradio-container { padding: 0.55rem 0.6rem 0.85rem !important; }
  .form, form { gap: 0.45rem !important; }
  .poem-genre-row { flex-wrap: wrap !important; gap: 0.3rem !important; }
  .poem-genre-row > * { flex: 1 1 calc(33.33% - 0.3rem) !important; min-width: 4.5rem !important; }
  .poem-slider-row { flex-direction: column !important; }
  .poem-slider-row > * { width: 100% !important; min-width: 0 !important; }
  #poem-in textarea { min-height: 2.75rem !important; max-height: 24vh !important; }
  #poem-out textarea { min-height: 5rem !important; max-height: min(36vh, 220px) !important; }
  .poem-submit-btn { width: 100% !important; }
}
@media (max-width: 480px) {
  .prose h2 { font-size: 1.2rem !important; margin: 0.35em 0 !important; }
  .prose h3, .prose h4 { font-size: 0.95rem !important; margin: 0.35em 0 !important; }
}
"""


def _get_subproject_infer(project_key: str, subdir: str):
    mod_name = f"_course_lab_infer_{project_key}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    project_dir = (ROOT / subdir).resolve()
    infer_py = project_dir / "infer.py"
    if not infer_py.is_file():
        raise FileNotFoundError(infer_py)

    sys.modules.pop("model", None)
    path_str = str(project_dir)
    sys.path.insert(0, path_str)
    try:
        spec = importlib.util.spec_from_file_location(mod_name, infer_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {infer_py}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if sys.path and sys.path[0] == path_str:
            sys.path.pop(0)
        sys.modules.pop("model", None)


def poem_ui(prompt: str, genre_label: str, max_chars: int, temperature: float):
    try:
        infer_mod = _get_subproject_infer("poem", "吟诗作赋")
        body = (prompt or "").strip() or DEFAULT_PROMPT
        label = (genre_label or "不限").strip() or "不限"
        if label not in GENRE_CHARS_PER_CLAUSE:
            label = "不限"
        n_per = GENRE_CHARS_PER_CLAUSE[label]

        text = infer_mod.generate_lines(
            body,
            ckpt_path=CKPT_POEM if CKPT_POEM.is_file() else None,
            max_new_chars=int(max_chars),
            temperature=float(temperature),
            chars_per_clause=n_per,
        )
        return text, "完成"
    except Exception as e:
        return "", f"错误: {e}\n请先运行: python 吟诗作赋/train.py"


def _genre_click(label: str) -> tuple[str, str, dict, str]:
    ex = EXAMPLES_BY_GENRE.get(label, EXAMPLES_BY_GENRE["不限"])
    first = ex[0] if ex else DEFAULT_PROMPT
    dd = gr.update(choices=ex, value=first)
    info = GENRE_INFO_LABEL.get(label, label)
    return label, info, dd, first


def _apply_example_prompt(choice: str) -> str:
    return choice if choice else DEFAULT_PROMPT


def build_demo():
    with gr.Blocks(
        title="计算智能导论 — 吟诗作赋",
    ) as demo:
        gr.Markdown(
            "## 吟诗作赋 · 古诗续写\n"
            "字符级 LSTM，在上句或提示后续写。"
        )
        with gr.Accordion("使用说明（点击展开）", open=False):
            gr.Markdown(
                "- 句末加「，」「。」等停顿，续写更稳。\n"
                "- 七言/五言会约束每小句 **7 / 5** 个汉字（不含标点）；点体裁换示例。\n"
                "- 续写最多 **300** 字；温度越高越随机。\n"
                "- 首次请在 `吟诗作赋` 目录运行 `python train.py` 生成权重。"
            )
        gr.Markdown("#### 体裁与示例")

        genre_state = gr.State("不限")
        genre_info = gr.Textbox(
            label="当前体裁",
            value=GENRE_INFO_LABEL["不限"],
            interactive=False,
            lines=1,
        )
        p_example = gr.Dropdown(
            choices=EXAMPLES_BY_GENRE["不限"],
            value=DEFAULT_PROMPT,
            label="示例（点体裁后切换两条）",
        )
        p_in = gr.Textbox(
            label="主题 / 上句提示",
            lines=2,
            value=DEFAULT_PROMPT,
            placeholder="例：床前明月光，（前半句 + 逗号）",
            elem_id="poem-in",
        )
        p_example.change(_apply_example_prompt, [p_example], [p_in])

        with gr.Row(elem_classes=["poem-genre-row"]):
            for _lbl in ("不限", "七言", "五言", "词", "乐府"):
                bb = gr.Button(_lbl, size="sm")
                bb.click(
                    lambda l=_lbl: _genre_click(l),
                    None,
                    [genre_state, genre_info, p_example, p_in],
                )

        with gr.Row(elem_classes=["poem-slider-row"]):
            p_chars = gr.Slider(10, 300, value=200, step=10, label="续写最大长度")
            p_temp = gr.Slider(0.5, 1.5, value=0.9, step=0.05, label="温度")

        p_out = gr.Textbox(label="生成文本", lines=8, elem_id="poem-out")
        p_msg = gr.Textbox(label="状态", lines=1, max_lines=2)

        gr.Button("生成", variant="primary", elem_classes=["poem-submit-btn"]).click(
            poem_ui, [p_in, genre_state, p_chars, p_temp], [p_out, p_msg]
        )

    return demo


def _guess_lan_ipv4() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        hn = socket.gethostname()
        for info in socket.getaddrinfo(hn, None, socket.AF_INET, socket.SOCK_STREAM):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return None


if __name__ == "__main__":
    _host = (os.environ.get("GRADIO_SERVER_NAME") or "0.0.0.0").strip()
    _port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    _quiet = _host in ("0.0.0.0", "::", "[::]")
    _demo = build_demo()
    _, _local_url, _ = _demo.launch()
    if _quiet:
        _p = urlparse(_local_url).port or _port
        _lan = _guess_lan_ipv4()
        print("\n--- 复制到浏览器即可 ---")
        print(f"本机:   http://127.0.0.1:{_p}/")
        if _lan:
            print(f"局域网: http://{_lan}:{_p}/")
        else:
            print("局域网: （未能自动检测 IPv4，请在本机网络设置里查看地址后手动替换）")
        print("（同网段访问须放行防火墙入站；若检测到的 IP 不对，多半是 VPN/多网卡，请以系统显示为准。）\n")
