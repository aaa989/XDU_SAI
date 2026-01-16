import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义Rosenbrock函数
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

# 定义梯度函数
def gradient_rosenbrock(x):
    grad_x0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    grad_x1 = 200 * (x[1] - x[0] ** 2)
    return np.array([grad_x0, grad_x1])

# DFP拟牛顿法
def dfp_method(func, grad_func, x0, max_iter=1000, tol=1e-6):
    n = len(x0)
    H = np.eye(n)
    x = x0
    for k in range(max_iter):
        g = grad_func(x)
        d = -np.dot(H, g)
        alpha = line_search(func, x, d)
        x_new = x + alpha * d
        s = x_new - x
        y = grad_func(x_new) - g
        rho = 1 / np.dot(y, s)
        A = np.eye(n) - rho * np.outer(s, y)
        B = np.eye(n) - rho * np.outer(y, s)
        H = np.dot(A, np.dot(H, B)) + rho * np.outer(s, s)
        x = x_new
        if np.linalg.norm(grad_func(x)) < tol:
            break
    return x, k

# 一维线性搜索（简单的回溯法）
def line_search(func, x, d, c1=1e-4, rho=0.5):
    alpha = 1
    while func(x + alpha * d) > func(x) + c1 * alpha * np.dot(gradient_rosenbrock(x), d):
        alpha *= rho
    return alpha

# 选取初始点
x0 = np.array([-1.2, 1])
x_opt, num_iter = dfp_method(rosenbrock, gradient_rosenbrock, x0)

# 生成网格点用于绘图
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-1, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros(X1.shape)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = rosenbrock([X1[i, j], X2[i, j]])

# 创建一个包含两个子图的窗口
fig = plt.figure(figsize=(12, 6))  # 调整整体画布大小

# 绘制3D图（交换X1和X2的位置）
ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # 改为横向排列，1行2列的第1个
ax1.plot_surface(X2, X1, Z, cmap='rainbow')  # 交换X1和X2的位置
ax1.set_xlabel('x2')  # 交换后x2对应原来的x1轴
ax1.set_ylabel('x1')  # 交换后x1对应原来的x2轴
ax1.set_zlabel('f(x)')
ax1.set_title('Rosenbrock and path of Quasi-Newton (3D)')

# 绘制等高线图（保持原样）
ax2 = fig.add_subplot(1, 2, 2)  # 改为横向排列，1行2列的第2个
contour = ax2.contourf(X1, X2, Z, levels=50, cmap='rainbow')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Rosenbrock and path of Quasi-Newton (Contour)')
fig.colorbar(contour, ax=ax2)

plt.tight_layout()  # 自动调整子图间距，使布局更协调
plt.show()