import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Rosenbrock函数
def rosenbrock(x):
    return (x[0] ** 2 - x[1]) ** 2 + (1 - x[0]) ** 2


# Rosenbrock函数的梯度
def rosenbrock_grad(x):
    df_dx1 = 4 * x[0] * (x[0] ** 2 - x[1]) - 2 * (1 - x[0])
    df_dx2 = -2 * (x[0] ** 2 - x[1])
    return np.array([df_dx1, df_dx2])


# 最速下降法
def gradient_descent(x0, learning_rate, num_iterations):
    x = x0.copy()
    path = [x.copy()]
    for _ in range(num_iterations):
        grad = rosenbrock_grad(x)
        x -= learning_rate * grad
        path.append(x.copy())
    return np.array(path)


# 绘制Rosenbrock函数和梯度下降路径
def plot_rosenbrock_and_gradient_descent():
    # 创建网格
    x = np.linspace(-1.5, 2, 400)
    y = np.linspace(-0.5, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])  # 修正参数传递方式

    # 绘制3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6, edgecolor='none')

    # 梯度下降参数设置
    x0 = np.array([1.2, 1.2])
    learning_rate = 0.002  # 调整学习率
    num_iterations = 5000  # 增加迭代次数
    path = gradient_descent(x0, learning_rate, num_iterations)

    # 计算路径点的Z值
    path_z = rosenbrock(path.T)  # 转置保证形状匹配
    ax.plot(path[:, 0], path[:, 1], path_z, color='r', linewidth=1.5)

    plt.title('third question_1')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x)$')

    # 绘制2D等高线图
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='jet')
    plt.colorbar(label='Function Value')
    plt.plot(path[:, 0], path[:, 1], 'w-', linewidth=1.5)  # 白色路径更明显
    plt.title('third question_2')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.show()


# 执行绘图函数
plot_rosenbrock_and_gradient_descent()