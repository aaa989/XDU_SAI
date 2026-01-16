import numpy as np
from IPython.display import clear_output
import time

class Maze:
    def __init__(self):
        self.grid = np.array([
            ['S', ' ', ' ', ' ', ' ', 'W', ' '],
            [' ', ' ', 'T', ' ', ' ', 'T', ' '],
            [' ', 'W', ' ', 'W', ' ', 'W', ' '],
            ['T', 'W', ' ', ' ', ' ', 'W', ' '],
            [' ', ' ', 'T', 'W', ' ', ' ', ' '],
            [' ', 'T', ' ', ' ', 'T', ' ', ' '],
            [' ', 'W', ' ', ' ', ' ', ' ', 'R']
        ])
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.path_history = [self.agent_pos]
        return self.pos_to_state(self.agent_pos)

    def pos_to_state(self, pos):
        return pos[0] * 7 + pos[1]

    def step(self, action):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        new_pos = (self.agent_pos[0] + moves[action][0],
                   self.agent_pos[1] + moves[action][1])

        # 检查边界和墙
        if (not (0 <= new_pos[0] < 7)) or (not (0 <= new_pos[1] < 7)) or (self.grid[new_pos] == 'W'):
            return self.pos_to_state(self.agent_pos), -1, False

        self.agent_pos = new_pos
        self.path_history.append(new_pos)

        cell = self.grid[new_pos]
        if cell == 'T':
            return self.pos_to_state(new_pos), -5, True
        if cell == 'R':
            return self.pos_to_state(new_pos), 50, True
        return self.pos_to_state(new_pos), -0.1, False

def visualize(maze, q_table=None, show_path=False):
    grid_display = maze.grid.copy()
    y, x = maze.agent_pos
    grid_display[y, x] = 'A'

    print("当前迷宫：")
    for row in grid_display:
        print("  ".join(row))

    if q_table is not None:
        state = maze.pos_to_state((y, x))
        print(f"\n当前位置({y},{x})的Q值：")
        print("向右: {:.1f}".format(q_table[state, 0]),"向下: {:.1f}".format(q_table[state, 1]),
              "向左: {:.1f}".format(q_table[state, 2]),"向上: {:.1f}".format(q_table[state, 3]))

    if show_path and hasattr(maze, 'path_history'):
        print("\n移动路径：")
        path_str = ""
        for i, pos in enumerate(maze.path_history):
            if i > 0:
                path_str += " → "
            path_str += f"({pos[0]},{pos[1]})"
        print(path_str)
        print(f"共有: {len(maze.path_history) - 1}步")

# 训练参数设置
env = Maze()
best_reward = -float('inf')
best_path = []
q_table = np.zeros((49, 4))  # 7x7=49状态，4种动作
max_episodes = 10000
early_stop = False
optimal_threshold = 48.9  # 理论最大奖励=50-0.1*最优步数(约12步)
display_interval = 500  # 每500轮显示一次

# 训练过程
for episode in range(max_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # ε-greedy策略（随时间衰减）
        epsilon = max(0.01, 0.1 * (1 - episode / max_episodes))
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done = env.step(action)
        total_reward += reward

        # Q-learning更新
        q_table[state, action] += 0.1 * (
                reward + 0.9 * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state

    # 更新最佳奖励和路径
    if total_reward > best_reward:
        best_reward = total_reward
        best_path = env.path_history.copy()

    elif episode % display_interval == 0:
        # 定期显示当前训练情况
        clear_output(wait=True)
        print(f"训练进度: {episode}/{max_episodes} 轮")
        print(f"当前最佳奖励: {best_reward:.1f}")
        visualize(env, q_table, show_path=True)  # 显示路径
        time.sleep(0.2)

    # 提前终止条件
    if best_reward >= optimal_threshold:
        print(f"\n提前达标！在第 {episode} 轮学会最优路径")
        early_stop = True
        break

# 最终结果展示
if not early_stop:
    print(f"\n达到最大训练轮数 {max_episodes}")

# 使用最佳路径重置环境
env.reset()
for action in range(len(best_path) - 1):
    # 重建动作序列
    current = best_path[action]
    next_pos = best_path[action + 1]
    if next_pos[0] > current[0]:  # 向下
        env.step(1)
    elif next_pos[0] < current[0]:  # 向上
        env.step(3)
    elif next_pos[1] > current[1]:  # 向右
        env.step(0)
    elif next_pos[1] < current[1]:  # 向左
        env.step(2)

print("\n最终结果：")
visualize(env, q_table, show_path=True)  # 显示路径
print(f"最佳回合奖励: {best_reward:.1f}")