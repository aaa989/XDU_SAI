
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def calculate_probability(n):
# 计算至少有两人生日相同的概率
    if n > 365:
        return 1.0 # 鸽巢原理
    prob_unique = 1.0
    for i in range(n):
        prob_unique *= (365 - i) / 365.0 # 通过这个实现组合公式
    return 1 - prob_unique  # 1-至少两个人同一天生日＝不重一天过生日概率


def update_plot(n):
    # 更新图表数据
    probs = [calculate_probability(i) for i in range(1, n + 1)]
    ax.clear()
    ax.plot(range(1, n + 1), probs, label=f'People ={n}')
    ax.set_xlabel('Number of People')
    ax.set_ylabel('Probability')
    ax.set_title('Birthday Paradox Visualization')
    ax.legend()
    ax.grid(True)
    canvas.draw()


def on_value_change(event=None):
# 当用户改变人数时更新图表和概率显示
    n = int(entry.get())
    prob_label.config(text=f'Probability: {calculate_probability(n):.8f}')
    update_plot(n)


# 创建Tkinter窗口
root = tk.Tk()
root.title('生日谬论演示')

# 创建输入框和标签
ttk.Label(root, text="Enter number of people:").pack(padx=300, pady=5)
entry = ttk.Entry(root)
entry.pack(padx=10, pady=5)
entry.bind('<Return>', on_value_change)
entry.bind('<FocusOut>', on_value_change) # 当输入框失去焦点时也触发更新

# 创建显示概率的标签
prob_label = ttk.Label(root, text="")
prob_label.pack(padx=10, pady=5)

# 创建matplotlib图表
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root) # 将matplotlib图表嵌入到Tkinter窗口中
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 初始化图表显示
update_plot(23) # 默认显示23人的概率

# 运行Tkinter主循环
root.mainloop()