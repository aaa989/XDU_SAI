# 基于深度学习的人脸识别系统

一个基于 PyTorch 和 Flask 的人脸识别项目，支持实时摄像头识别、静态图片识别和人脸底库管理功能。

## 功能特性

- 🎥 **实时人脸识别**：通过摄像头实时检测和识别人脸
- 🖼️ **静态图片识别**：上传图片进行离线人脸识别
- 📁 **人脸底库管理**：支持添加、删除、列出已注册的人脸
- 📋 **识别记录日志**：记录最近的识别记录
- 🌐 **Web 界面**：简洁美观的网页界面，易于操作

## 技术栈

- **框架**: Flask 2.x
- **深度学习**: PyTorch 1.x
- **人脸检测**: MTCNN (facenet-pytorch)
- **人脸识别**: InceptionResnetV1 (facenet-pytorch)
- **前端**: HTML5 + CSS3 + JavaScript
- **图像处理**: OpenCV, PIL

## 快速开始

### 1. 环境要求

```bash
Python 3.8+
PyTorch 1.8+
```

### 2. 安装依赖

```bash
# 进入项目目录
cd 人脸识别/FaceRecognitionProject

# 安装依赖
pip install -r requirements.txt
```

如果没有 `requirements.txt`，可以手动安装：

```bash
pip install flask torch torchvision facenet-pytorch opencv-python pillow numpy
```

### 3. 启动服务

```bash
python main.py
```

启动成功后，控制台会显示：

```
==================================================
智能人脸识别后端已启动
请打开浏览器访问: http://127.0.0.1:5000
==================================================
```

### 4. 访问系统

打开浏览器，访问 `http://127.0.0.1:5000` 即可使用人脸识别系统。

## 使用说明

### 1. 动态识别（摄像头）

1. 点击「动态识别」标签页
2. 点击「开启摄像头」按钮
3. 系统会实时检测并识别人脸
4. 识别结果会显示在右侧的识别记录列表中

### 2. 静态识别（图片）

1. 点击「静态识别」标签页
2. 点击「选择图片」上传一张包含人脸的图片
3. 点击「开始识别」按钮
4. 识别结果会显示在右侧列表中

### 3. 底库管理

1. 点击「底库管理」标签页
2. **本地上传**：点击「本地上传」按钮，输入姓名并选择图片
3. **拍照录入**：点击「拍照录入」按钮，使用摄像头拍照并框选人脸区域
4. **删除人员**：点击表格右侧的「删除」按钮移除已注册的人脸

## 项目结构

```
计算机视觉大作业/
├── 人脸识别/
│   ├── FaceRecognitionProject/       # Web 应用主目录
│   │   ├── face_database/            # 人脸图片数据库
│   │   ├── templates/                # HTML 模板
│   │   │   └── index.html            # 主页面
│   │   ├── face_features.pkl         # 人脸特征数据文件
│   │   └── main.py                   # Flask 应用入口
│   └── xunlian/                      # 模型训练脚本目录
│       ├── xunlian.py                # ArcFace 训练脚本
│       └── my_lfw_resnet18_weights.pth  # 训练好的模型权重
├── 测试/                             # 测试图片
├── 项目报告.docx                     # 项目报告
└── 基于深度学习的人脸识别项目汇报.pptx  # 汇报 PPT
```

## API 接口

| 接口                    | 方法 | 说明             |
| ----------------------- | ---- | ---------------- |
| `/api/camera/start`     | POST | 开启摄像头       |
| `/api/camera/stop`      | POST | 关闭摄像头       |
| `/api/camera/logs`      | GET  | 获取识别记录     |
| `/api/recognize_static` | POST | 静态图片识别     |
| `/api/db/list`          | GET  | 获取底库列表     |
| `/api/db/add`           | POST | 添加人脸到数据库 |
| `/api/db/delete`        | POST | 从数据库删除人脸 |

## 注意事项

1. 首次运行时，系统会自动下载 MTCNN 和 InceptionResnetV1 模型（约 100MB）
2. 建议使用 Chrome 或 Edge 浏览器以获得最佳体验
3. 摄像头功能需要在真实浏览器中使用，IDE 内置浏览器可能不支持
4. 确保上传的人脸图片清晰，正脸朝向镜头

## 许可证

本项目仅供学习和研究使用。

---

**Powered by PyTorch & Facenet**
