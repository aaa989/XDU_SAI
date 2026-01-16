import numpy as np
import matplotlib.pyplot as plt


def golden_section_search(f, a, b, tol=0.001):
    """黄金分割法求函数极小值"""
    gr = (np.sqrt(5) - 1) / 2  # 黄金比例
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    steps = 0

    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        steps += 1

    x_min = (a + b) / 2
    return x_min, f(x_min), steps


# 定义目标函数
f = lambda x: x ** 3 - 4 * x - 1

# 求解极小值
x_min, f_min, steps = golden_section_search(f, 0, 3)

# 绘制函数曲线
x = np.linspace(0, 3, 100)
plt.plot(x, f(x), 'b-', label='f(x) = $x^3 - 4x - 1$')
plt.scatter(x_min, f_min, c='red', label=f'Minimum: ({x_min:.4f}, {f_min:.4f})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('second question')
plt.legend()
plt.grid()
plt.show()

# 输出结果
print(f"极小值点: {x_min:.4f}")
print(f"极小值: {f_min:.4f}")
print(f"迭代次数: {steps}")