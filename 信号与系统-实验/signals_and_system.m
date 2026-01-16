% 读取图像（保持彩色）
originalImg = imread('D:\app\基本组件\Desktop\实验图像.jpeg');

% 高斯平滑滤波
sigma = 1; % 控制平滑程度，n越大图越模糊
filterSize = 3; % 滤波器大小（奇数），简单来说就是调整是n×n卷积
smoothedColorImg = imgaussfilt(originalImg, sigma, 'FilterSize', filterSize);

% Canny边缘检测
grayImg = rgb2gray(originalImg); 
lowThreshold = 0.08; % 低阈值，低于0.08视作噪声点，不保留
highThreshold = 0.2; % 高阈值，高于0.2视作边缘，被保留下来
edgeImg = edge(grayImg, 'canny', [lowThreshold highThreshold], sqrt(2));% 把找出的边缘点留下，45°范围内的连接在一起

% 结果可视化
figure;
subplot(1,3,1), imshow(originalImg), title('原图');
subplot(1,3,2), imshow(smoothedColorImg), title('平滑后');
subplot(1,3,3), imshow(edgeImg), title('边缘提取');

% 保存结果
imwrite(smoothedColoImg, 'smoothed.jpg');
imwrite(edgeImg, 'edges.png');