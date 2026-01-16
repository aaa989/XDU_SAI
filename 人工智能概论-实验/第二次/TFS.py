import numpy as np
import random

# 城市间的距离矩阵（对称矩阵，对角线为0）
# 示例中包含了10个城市之间的距离
city_distances = np.array([
    [0, 93, 49, 51, 46, 58, 93, 68, 69, 127],
    [93, 0, 45, 52, 111, 141, 67, 25, 154, 90],
    [49, 45, 0, 28, 69, 97, 72, 21, 114, 105],
    [51, 52, 28, 0, 86, 108, 46, 29, 102, 80],
    [46, 111, 69, 86, 0, 33, 132, 90, 100, 167],
    [58, 141, 97, 108, 33, 0, 152, 118, 86, 186],
    [93, 67, 72, 46, 132, 152, 0, 60, 124, 34],
    [68, 25, 21, 29, 90, 118, 60, 0, 128, 90],
    [69, 154, 114, 102, 100, 86, 124, 128, 0, 151],
    [127, 90, 105, 80, 167, 186, 34, 90, 151, 0]
])

# 遗传算法参数
POP_SIZE = 100  # 种群大小，即有多少条不同的路径在同时进化
GENES = list(range(10))  # 基因（城市编号），表示10个城市
generation = 10000  # 迭代次数，即算法运行多少代
MUTATION_RATE = 0.01  # 变异率，控制每个基因变异的可能性
cross_rate = 0.8  # 交叉率，控制两个父代交叉产生子代的可能性


# 初始化种群
# 创建初始种群，每个个体都是一个随机的城市排列
def create_population(pop_size, genes):
    return [random.sample(genes, len(genes)) for _ in range(pop_size)]


population = create_population(POP_SIZE, GENES)


# 适应度函数
# 计算给定路径的总距离
def fitness(every):
    total_distance = sum(city_distances[every[i]][every[i + 1]] for i in range(len(every) - 1))
    total_distance += city_distances[every[-1]][every[0]]  # 回到起点
    return total_distance


# 选择操作（轮盘赌选择）
# 根据适应度概率选择个体，适应度越低（距离越短）被选中的概率越高
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]  # 计算每个个体的选择概率
    selected_index = np.random.choice(range(len(population)), p=probabilities)  # 根据概率选择个体
    return population[selected_index]


# 交叉操作（单点交叉）
# 两个父代个体交换部分基因，产生两个新的子代个体
def crossover(parent1, parent2):
    if random.random() < cross_rate:  # 根据交叉率决定是否进行交叉
        point = random.randint(1, len(parent1) - 2)  # 随机选择一个交叉点
        # 生成两个新的子代，确保没有重复的基因
        child1 = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
        child2 = parent2[:point] + [gene for gene in parent1 if gene not in parent2[:point]]
        return child1, child2
    else:
        return parent1, parent2  # 不进行交叉，直接返回父代


# 变异操作（随机交换两个基因）
# 随机选择两个基因并交换它们的位置
def mutate(every):
    if random.random() < MUTATION_RATE:  # 根据变异率决定是否进行变异
        i, j = random.sample(range(len(every)), 2)  # 随机选择两个基因的位置
        every[i], every[j] = every[j], every[i]  # 交换两个基因
    return every


# 遗传算法主循环
best_way = None  # 记录最优解
best_match = float('inf')  # 记录最优解的适应度（初始化为正无穷大）

for generation in range(generation):  # 迭代N代
    fitnesses = [fitness(ind) for ind in population]  # 计算当前种群中每个个体的适应度
    new_population = []  # 存储新种群的列表

    # 进行选择、交叉和变异操作，生成新的种群
    for _ in range(POP_SIZE // 2):  # 每次生成两个新的个体
        parent1 = select(population, fitnesses)  # 选择一个父代个体
        parent2 = select(population, fitnesses)  # 选择另一个父代个体
        child1, child2 = crossover(parent1, parent2)  # 交叉产生两个子代个体
        new_population.append(mutate(child1))  # 对第一个子代进行变异并加入新种群
        new_population.append(mutate(child2))  # 对第二个子代进行变异并加入新种群

    population = new_population  # 更新种群

    # 记录当前代最优解
    current_best_way_every = min(population, key=fitness)  # 找到当前种群中适应度最低的个体
    current_best_match = fitness(current_best_way_every)  # 计算其适应度

    # 如果当前最优解的适应度比历史最优解的适应度更低，则更新最优解
    if current_best_match < best_match:
        best_match = current_best_match
        best_way = current_best_way_every

    # 输出当前代最优解和对应适应度（可用于观察算法进化过程）
    print(f"Generation {generation}: best_way Fitness = {current_best_match}, best_way Path = {current_best_way_every}")

# 输出最终结果
print(f"迭代次数为： {generation+1}，最佳方案: 路径为 {best_way}, 总路程 = {best_match}")