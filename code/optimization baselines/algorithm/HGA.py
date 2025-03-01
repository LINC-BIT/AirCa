
import numpy as np
import random
from deap import base, creator, tools, algorithms


def cargo_load_planning_genetic(weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u,
                                max_positions, population_size=100, generations=100, crossover_prob=0.7, mutation_prob=0.2):
    """
    使用改进版遗传算法计算货物装载方案，最小化重心的变化量。

    参数:
        weights (list): 每个货物的质量列表。
        cargo_names (list): 每个货物的名称。
        cargo_types_dict (dict): 货物名称和占用的货位数量。
        positions (list): 可用的货位编号。
        cg_impact (list): 每个位置每kg货物对重心index的影响系数。
        cg_impact_2u (list): 两个位置组合的重心影响系数。
        cg_impact_4u (list): 四个位置组合的重心影响系数。
        max_positions (int): 总货位的数量。
        population_size (int): 遗传算法的种群大小。
        generations (int): 遗传算法的代数。
        crossover_prob (float): 交叉操作的概率。
        mutation_prob (float): 变异操作的概率。

    返回:
        best_solution (np.array): 最优装载方案矩阵。
        best_cg_change (float): 最优方案的重心变化量。
    """
    try:
        # 将货物类型映射为对应的占用单位数
        cargo_types = [cargo_types_dict[name] for name in cargo_names]

        num_cargos = len(weights)       # 货物数量
        num_positions = len(positions)  # 可用货位数量

        # 定义适应度函数（最小化重心变化量）
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 目标是最小化
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # 个体初始化函数
        def init_individual():
            individual = []
            occupied = [False] * num_positions
            for cargo_type in cargo_types:
                if cargo_type == 1:
                    valid_positions = [j for j in range(num_positions) if not occupied[j]]
                elif cargo_type == 2:
                    valid_positions = [j for j in range(0, num_positions - 1, 2) if not any(occupied[j + k] for k in range(cargo_type))]
                elif cargo_type == 4:
                    valid_positions = [j for j in range(0, num_positions - 3, 4) if not any(occupied[j + k] for k in range(cargo_type))]
                else:
                    valid_positions = []

                if not valid_positions:
                    # 如果没有有效位置，随机选择一个符合类型对齐的起始位置
                    if cargo_type == 1:
                        start_pos = random.randint(0, num_positions - 1)
                    elif cargo_type == 2:
                        choices = [j for j in range(0, num_positions - 1, 2)]
                        if choices:
                            start_pos = random.choice(choices)
                        else:
                            start_pos = 0  # 默认位置
                    elif cargo_type == 4:
                        choices = [j for j in range(0, num_positions - 3, 4)]
                        if choices:
                            start_pos = random.choice(choices)
                        else:
                            start_pos = 0  # 默认位置
                    else:
                        start_pos = 0  # 默认位置
                else:
                    start_pos = random.choice(valid_positions)

                individual.append(start_pos)

                # 标记占用的位置
                for k in range(cargo_type):
                    pos = start_pos + k
                    if pos < num_positions:
                        occupied[pos] = True

            return creator.Individual(individual)

        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 适应度评估函数
        def evaluate(individual):
            # 检查重叠和边界
            occupied = [False] * num_positions
            penalty = 0
            cg_change = 0.0

            for i, start_pos in enumerate(individual):
                cargo_type = cargo_types[i]
                weight = weights[i]

                # 检查边界
                if start_pos < 0 or start_pos + cargo_type > num_positions:
                    penalty += 10000 # 超出边界的严重惩罚
                    continue

                # 检查重叠
                overlap = False
                for k in range(cargo_type):
                    pos = start_pos + k
                    if occupied[pos]:
                        penalty += 10000  # 重叠的严重惩罚
                        overlap = True
                        break
                    occupied[pos] = True
                if overlap:
                    continue

                # 计算重心变化量
                if cargo_type == 1:
                    cg_change += abs(weight * cg_impact[start_pos])
                elif cargo_type == 2:
                    if start_pos % 2 == 0 and (start_pos // 2) < len(cg_impact_2u):
                        cg_change += abs(weight * cg_impact_2u[start_pos // 2])
                    else:
                        penalty += 10000  # 不对齐的严重惩罚
                elif cargo_type == 4:
                    if start_pos % 4 == 0 and (start_pos // 4) < len(cg_impact_4u):
                        cg_change += abs(weight * cg_impact_4u[start_pos // 4])
                    else:
                        penalty += 10000  # 不对齐的严重惩罚

            return (cg_change + penalty,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxOnePoint)  # 改为单点交叉
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutation_prob)  # 使用交换变异
        toolbox.register("select", tools.selRoulette)  # 轮盘赌选择

        # 初始化种群
        population = toolbox.population(n=population_size)

        # 运行遗传算法
        try:
            algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=1.0, ngen=generations,
                                verbose=False)
        except ValueError as e:
            print(f"遗传算法运行时出错: {e}")
            return [], -1000000  # 返回空列表和一个负的重心变化量作为错误标志

        # 选择最优个体
        try:
            best_individual = tools.selBest(population, 1)[0]
            best_cg_change = evaluate(best_individual)[0]
        except IndexError as e:
            print(f"选择最优个体时出错: {e}")
            return [], -1000000  # 返回空列表和一个负的重心变化量作为错误标志

        # 构建装载方案矩阵
        solution = np.zeros((num_cargos, num_positions))
        for i, start_pos in enumerate(best_individual):
            cargo_type = cargo_types[i]
            for k in range(cargo_type):
                pos = start_pos + k
                if pos < num_positions:
                    solution[i, pos] = 1

        return solution, best_cg_change
    except Exception as e:
        print(f"发生错误: {e}")
        return [], -1000000


def main():
    # 示例输入
    weights = [500, 800, 1200, 300, 700, 1000, 600, 900]  # 每个货物的质量
    cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']  # 货物名称
    cargo_types_dict = {"LD3": 1, "PLA": 2, "P6P": 4, "BULK": 1}  # 货物占位关系
    positions = list(range(44))  # 44个货位编号
    cg_impact = [i * 0.1 for i in range(44)]  # 每kg货物对重心index的影响系数 (单个位置)
    cg_impact_2u = [i * 0.08 for i in range(22)]  # 两个位置组合的影响系数
    cg_impact_4u = [i * 0.05 for i in range(11)]  # 四个位置组合的影响系数
    max_positions = 44  # 总货位数量

    # 调用遗传算法进行货物装载优化
    best_solution, best_cg_change = cargo_load_planning_genetic(weights, cargo_names, cargo_types_dict, positions,
                                                                cg_impact, cg_impact_2u, cg_impact_4u, max_positions)

    if best_solution:
        print("最优装载方案矩阵：")
        print(best_solution)
        print(f"最优装载方案的重心变化量：{best_cg_change:.2f}")
    else:
        print("优化失败，请检查输入或算法设置。")

if __name__ == "__main__":
    main()

