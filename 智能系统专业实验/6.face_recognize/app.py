# 核心依赖导入
import base64
import cv2
import time
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from torchvision import transforms

# 项目内部模块导入
import utils
from model import MyModel

app = Flask(__name__, template_folder='.')

# 初始化模型与配置
print("正在加载配置文件与模型...")
config = utils.read_config("config.json")
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

# 加载剪枝后的表情识别模型
model = torch.load(config['save_path'] + "pruned_model_full.pth", map_location=device, weights_only=False)
model = model.to(device)
model.eval()

# 表情分类标签（与训练集/部署脚本保持一致）
label_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# 图像预处理流程（适配模型输入要求）
data_trans = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载OpenCV人脸检测分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 全局变量：摄像头实例、最新表情检测日志
camera = None
latest_emotions = []

# 视频流帧处理生成器：实时人脸检测+表情识别
def gen_frames():
    global camera, latest_emotions
    while camera is not None and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        
        # 人脸检测：灰度化+检测人脸区域
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        current_emotions = []
        for (x, y, w, h) in faces:
            # 绘制人脸框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 提取人脸ROI并转换格式
            face_roi = gray[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            face_pil = transforms.ToPILImage()(face_rgb)
            
            # 预处理后执行模型推理
            input_tensor = data_trans(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # 记录表情结果并绘制到画面
            emotion = label_names[pred]
            current_emotions.append({"emotion": emotion, "time": time.strftime("%H:%M:%S")})
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        # 更新最新表情日志
        latest_emotions = current_emotions

        # 编码帧为JPEG并生成视频流响应
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask路由：主页（返回前端页面）
@app.route('/')
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# 接口：开启摄像头
@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({"success": True})

# 接口：关闭摄像头（释放资源+清空日志）
@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global camera, latest_emotions
    if camera is not None:
        camera.release()
        camera = None
        latest_emotions = []
    return jsonify({"success": True})

# 接口：视频流推送（供前端img标签调用）
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 接口：获取最新表情检测日志（前端轮询）
@app.route('/api/camera/logs')
def get_logs():
    return jsonify({"logs": latest_emotions})

# 接口：静态图片上传识别表情
@app.route('/api/recognize_static', methods=['POST'])
def recognize_static():
    # 接收上传文件
    file = request.files.get('file')
    if not file:
        return jsonify({"success": False, "error": "未收到图片"}), 400

    # 转换文件为OpenCV格式
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # 人脸检测+表情识别（逻辑同视频流）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    emotions = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = gray[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        face_pil = transforms.ToPILImage()(face_rgb)
        
        input_tensor = data_trans(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        emotion = label_names[pred]
        emotions.append(emotion)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    # 编码结果图片为Base64返回前端
    _, buffer = cv2.imencode('.jpg', frame)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({"success": True, "image": base64_img, "emotions": emotions})

# 启动服务
if __name__ == '__main__':
    print("=======================================")
    print("后端服务已启动，请在浏览器中访问: http://127.0.0.1:5000")
    print("=======================================")
    app.run(host='0.0.0.0', port=5000, debug=False)