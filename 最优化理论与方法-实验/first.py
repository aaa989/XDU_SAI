import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# 定义线性规划问题
c = np.array([-2, -1])  # 最大化问题转换为最小化问题
A = np.array([
    [-2, -5],  # 2x1 + 5x2 >= 12 → -2x1 -5x2 <= -12
    [1, 2],    # x1 + 2x2 <= 8
    [-1, 0],   # x1 >= 0 → -x1 <= 0
    [0, -1],   # x2 >= 0 → -x2 <= 0
    [1, 0],    # x1 <= 4
    [0, 1]     # x2 <= 3
])
b = np.array([-12, 8, 0, 0, 4, 3])

# 求解线性规划问题
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 4), (0, 3)])

# 打印结果
print(f'最优值: {-res.fun:.2f}')
print(f'最优解: x1 = {res.x[0]:.2f}, x2 = {res.x[1]:.2f}')

# 绘制可行域
x1 = np.linspace(0, 6, 500)  # 横坐标扩展到6
x2_1 = (12 - 2*x1) / 5       # 2x1 + 5x2 = 12
x2_2 = (8 - x1) / 2           # x1 + 2x2 = 8
x2_3 = np.zeros_like(x1)      # x2 = 0
x2_4 = 3 * np.ones_like(x1)   # x2 = 3

plt.figure(figsize=(10, 7))  # 调整画布尺寸

# 绘制约束线
valid_mask = (x2_1 >= -1) & (x2_1 <= 6)
plt.plot(x1[valid_mask], x2_1[valid_mask], 'b-', label='2x1 + 5x2 = 12')
plt.plot(x1, x2_2, 'g-', label='x1 + 2x2 = 8')

# 填充红色可行域
fill_mask = (x1 >= 0) & (x1 <= 4)  # 原始x1范围
plt.fill_between(x1[fill_mask],
                 np.maximum(x2_3[fill_mask], x2_1[fill_mask]),
                 np.minimum(x2_4[fill_mask], x2_2[fill_mask]),
                 color='red', alpha=0.3, label='Region')

# 绘制多条目标函数等值线
for z in [2.4, 6, -res.fun]:
    x2_target = (z - 2*x1) / 1
    plt.plot(x1, x2_target, '--', alpha=0.5, label=f'z={z:.1f}')

# 标注最优解
plt.scatter(res.x[0], res.x[1], color='red', s=100, label='best')
plt.annotate(f'best: ({res.x[0]:.2f}, {res.x[1]:.2f})\nMax: {-res.fun:.2f}',
             xy=(res.x[0], res.x[1]),
             xytext=(res.x[0]-2, res.x[1]+0.5),
             arrowprops=dict(arrowstyle="->"))

# 设置图形属性
plt.xlim(0, 6)
plt.ylim(0, 4)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('first question')
plt.legend(loc='upper right', bbox_to_anchor=(1.32, 1))  # 微调图例位置
plt.grid(True)

plt.tight_layout()
plt.show()