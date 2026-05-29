# 吟诗作赋 - 字符级 LSTM 古诗续写

基于字符级 LSTM 的古诗续写模型，支持七言、五言等体裁约束。

## 项目结构

```
.
├── app_gradio.py          # Gradio 网页界面
├── lab_paths.py           # 路径配置
├── train_utils.py         # 训练工具函数
├── 吟诗作赋/
│   ├── train.py           # 模型训练脚本
│   ├── model.py           # CharLSTM 模型定义
│   ├── infer.py           # 推理脚本
│   ├── corpus.py          # 语料处理
│   └── checkpoints/       # 模型权重存放目录
│       └── char_lstm.pt   # 训练后的模型权重
└── course_lab_reference-main/
    └ course_lab_data/
        └ poem_generation/
            └ chinese-poetry/  # 古诗语料数据（全唐诗、宋词等）
```

## 快速启动

### 1. 安装依赖

```bash
pip install torch gradio tqdm
```

如需 GPU 训练，请安装 CUDA 版 PyTorch：

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### 2. 训练模型

首次使用需要训练模型生成权重文件：

```bash
cd 吟诗作赋
python train.py
```

训练参数（可选）：

```bash
python train.py --epochs 12 --batch_size 256 --device cuda:0
```

训练完成后，权重保存在 `吟诗作赋/checkpoints/char_lstm.pt`。

### 3. 启动 Gradio 界面

```bash
python app_gradio.py
```

启动后终端会显示访问地址：

- 本机访问：`http://127.0.0.1:7860/`
- 局域网访问：终端自动显示本机 IP

### 4. 命令行推理

```bash
cd 吟诗作赋
python infer.py --prompt "春眠不觉晓，" --chars 160 --temp 0.85
```

参数说明：

- `--prompt`: 开头提示词
- `--chars`: 生成字数
- `--temp`: 温度（越高越随机，建议 0.8-1.0）
- `--clause`: 七言设为 7，五言设为 5

## 功能说明

- **体裁约束**：支持七言（每句7字）、五言（每句5字）、词、乐府等
- **续写模式**：输入开头句，模型自动续写
- **局域网访问**：支持手机/其他电脑访问

## 断点续训

训练中断后可继续：

```bash
python train.py --resume
```

## 注意事项

1. Windows 防火墙需放行入站连接才能局域网访问
2. 若网页打不开，检查系统代理是否拦截了 `127.0.0.1`
3. 语料数据较大，首次训练需一定时间
