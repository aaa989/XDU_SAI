import cv2
import torch
import numpy as np
from torchvision import transforms
from model import MyModel
import utils

def main():
    # 配置与设备初始化
    config = utils.read_config("config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载表情识别模型
    model = torch.load(config['save_path']+"pruned_model_full.pth", map_location=device, weights_only=False)
    model.to(device).eval()

    label_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    
    # 图像预处理
    data_trans = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 人脸检测
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
            # 人脸预处理
            face_rgb = cv2.cvtColor(gray[y:y+h,x:x+w], cv2.COLOR_GRAY2RGB)
            input_tensor = data_trans(transforms.ToPILImage()(face_rgb)).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                pred = torch.argmax(model(input_tensor),dim=1).item()
            
            cv2.putText(frame,label_names[pred],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)

        cv2.imshow('Face Emotion Recognition',frame)
        if cv2.waitKey(1)&0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()