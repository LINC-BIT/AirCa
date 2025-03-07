# import numpy as np
# import pyswarms as ps
#
# def cargo_load_planning_pso_v2(weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u,
#                                max_positions, options=None, swarmsize=100, maxiter=100):
#     """
#     使用二进制粒子群优化方法计算货物装载方案，最小化重心的变化量。
#
#     参数:
#         weights (list): 每个货物的质量列表。
#         cargo_names (list): 每个货物的名称。
#         cargo_types_dict (dict): 货物名称和占用的货位数量。
#         positions (list): 可用的货位编号。
#         cg_impact (list): 每个位置每kg货物对重心index的影响系数。
#         cg_impact_2u (list): 两个位置组合的重心影响系数。
#         cg_impact_4u (list): 四个位置组合的重心影响系数。
#         max_positions (int): 总货位的数量。
#         options (dict, optional): PSO算法的配置选项。
#         swarmsize (int, optional): 粒子群大小。
#         maxiter (int, optional): 最大迭代次数。
#
#     返回:
#         best_solution (np.array): 最优装载方案矩阵。
#         best_cg_change (float): 最优方案的重心变化量。
#     """
#     # 将货物类型映射为对应的占用单位数
#     cargo_types = [cargo_types_dict[name] for name in cargo_names]
#
#     num_cargos = len(weights)  # 货物数量
#     num_positions = len(positions)  # 可用货位数量
#     dimension = num_cargos * max_positions  # 每个粒子的维度：货物数量 × 可用货位数量
#
#     # 如果未提供options，使用默认配置
#     if options is None:
#         options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
#
#     # 定义适应度评估函数
#     def fitness_function(x):
#         """
#         计算每个粒子的适应度值。
#
#         参数:
#             x (numpy.ndarray): 粒子的位置数组，形状为 (n_particles, dimension)。
#
#         返回:
#             numpy.ndarray: 每个粒子的适应度值。
#         """
#         fitness = np.zeros(x.shape[0])
#
#         for idx, particle in enumerate(x):
#             # 将连续位置映射为离散起始位置
#             start_positions = []
#             penalty = 0
#             cg_change = 0.0
#             occupied = np.zeros(num_positions, dtype=int)
#
#             for i in range(num_cargos):
#                 cargo_type = cargo_types[i]
#                 pos_continuous = particle[i * max_positions:(i + 1) * max_positions]
#
#                 # 根据粒子位置值选择最佳货位
#                 start_pos = np.argmax(pos_continuous)
#
#                 # 检查边界
#                 if start_pos < 0 or start_pos + cargo_type > num_positions:
#                     penalty += 1000
#                     continue
#
#                 # 检查对齐
#                 if cargo_type == 2 and start_pos % 2 != 0:
#                     penalty += 1000
#                 if cargo_type == 4 and start_pos % 4 != 0:
#                     penalty += 1000
#
#                 # 检查重叠
#                 if np.any(occupied[start_pos:start_pos + cargo_type]):
#                     penalty += 1000
#                 else:
#                     occupied[start_pos:start_pos + cargo_type] = 1
#
#                 start_positions.append(start_pos)
#
#                 # 计算重心变化量
#                 if cargo_type == 1:
#                     cg_change += weights[i] * cg_impact[start_pos]
#                 elif cargo_type == 2:
#                     cg_change += weights[i] * cg_impact_2u[start_pos // 2]
#                 elif cargo_type == 4:
#                     cg_change += weights[i] * cg_impact_4u[start_pos // 4]
#
#             fitness[idx] = cg_change + penalty
#
#         return fitness
#
#     # 设置PSO的边界
#     # 对于每个货物，起始位置的范围根据货物类型对齐
#     lower_bounds = []
#     upper_bounds = []
#     for i in range(num_cargos):
#         cargo_type = cargo_types[i]
#         lower_bounds.append([0] * max_positions)
#         upper_bounds.append([1] * max_positions)
#
#     bounds = (np.array(lower_bounds), np.array(upper_bounds))
#
#     # 初始化PSO优化器
#     optimizer = ps.single.GlobalBestPSO(n_particles=swarmsize, dimensions=dimension, options=options, bounds=bounds)
#
#     # 运行PSO优化
#     best_cost, best_pos = optimizer.optimize(fitness_function, iters=maxiter)
#
#     # 将最佳位置映射为离散装载方案
#     best_start_positions = []
#     penalty = 0
#     cg_change = 0.0
#     occupied = np.zeros(num_positions, dtype=int)
#
#     for i in range(num_cargos):
#         cargo_type = cargo_types[i]
#         pos_continuous = best_pos[i * max_positions:(i + 1) * max_positions]
#
#         # 根据粒子位置值选择最佳货位
#         start_pos = np.argmax(pos_continuous)
#
#         # 检查边界
#         if start_pos < 0 or start_pos + cargo_type > num_positions:
#             penalty += 1000
#             best_start_positions.append(start_pos)
#             continue
#
#         # 检查对齐
#         if cargo_type == 2 and start_pos % 2 != 0:
#             penalty += 1000
#         if cargo_type == 4 and start_pos % 4 != 0:
#             penalty += 1000
#
#         # 检查重叠
#         if np.any(occupied[start_pos:start_pos + cargo_type]):
#             penalty += 1000
#         else:
#             occupied[start_pos:start_pos + cargo_type] = 1
#
#         best_start_positions.append(start_pos)
#
#         # 计算重心变化量
#         if cargo_type == 1:
#             cg_change += abs(weights[i] * cg_impact[start_pos])
#         elif cargo_type == 2:
#             cg_change += abs(weights[i] * cg_impact_2u[start_pos // 2])
#         elif cargo_type == 4:
#             cg_change += abs(weights[i] * cg_impact_4u[start_pos // 4])
#
#     total_cg_change = cg_change + penalty
#
#     # 构建装载方案矩阵
#     best_xij = np.zeros((num_cargos, num_positions), dtype=int)
#     for i, start_pos in enumerate(best_start_positions):
#         cargo_type = cargo_types[i]
#         for k in range(cargo_type):
#             pos = start_pos + k
#             if pos < num_positions:
#                 best_xij[i, pos] = 1
#
#     # 检查是否有严重惩罚，判断是否找到可行解
#     if total_cg_change >= -999999:
#         return best_xij, total_cg_change
#     else:
#         return [], -1000000
#
#
# # 示例输入
# weights = [500, 800, 1200, 300, 700, 1000, 600, 900]  # 每个货物的质量
# cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']  # 货物名称
# cargo_types_dict = {"LD3": 1, "PLA": 2, "P6P": 4, "BULK": 1}  # 货物占位关系
# positions = list(range(44))  # 44个货位编号
# cg_impact = [i * 0.1 for i in range(44)]  # 每kg货物对重心index的影响系数 (单个位置)
# cg_impact_2u = [i * 0.08 for i in range(22)]  # 两个位置组合的影响系数
# cg_impact_4u = [i * 0.05 for i in range(11)]  # 四个位置组合的影响系数
# max_positions = 44  # 总货位数量
#
# # PSO算法配置选项
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
#
# # 调用粒子群优化进行货物装载优化
# best_solution, best_cg_change = cargo_load_planning_pso_v2(weights, cargo_names, cargo_types_dict, positions,
#                                                            cg_impact, cg_impact_2u, cg_impact_4u, max_positions,
#                                                            options=options, swarmsize=100, maxiter=100)
#
# # 输出结果
# if best_solution.size > 0:
#     print("最优装载方案矩阵：")
#     print(best_solution)
#     print(f"最优装载方案的重心变化量：{best_cg_change:.2f}")
# else:
#     print("优化失败，未找到可行解。")
import numpy as np
import pyswarms as ps


def cargo_load_planning_pso(weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u,
                            max_positions, options=None, swarmsize=100, maxiter=100):
    """
    使用粒子群优化方法计算货物装载方案，最小化重心的变化量。

    参数:
        weights (list): 每个货物的质量列表。
        cargo_names (list): 每个货物的名称。
        cargo_types_dict (dict): 货物名称和占用的货位数量。
        positions (list): 可用的货位编号。
        cg_impact (list): 每个位置每kg货物对重心index的影响系数。
        cg_impact_2u (list): 两个位置组合的重心影响系数。
        cg_impact_4u (list): 四个位置组合的重心影响系数。
        max_positions (int): 总货位的数量。
        options (dict, optional): PSO算法的配置选项。
        swarmsize (int, optional): 粒子群大小。
        maxiter (int, optional): 最大迭代次数。

    返回:
        best_solution (np.array): 最优装载方案矩阵。
        best_cg_change (float): 最优方案的重心变化量。
    """
    # 将货物类型映射为对应的占用单位数
    cargo_types = [cargo_types_dict[name] for name in cargo_names]

    num_cargos = len(weights)  # 货物数量
    num_positions = len(positions)  # 可用货位数量
    dimension = num_cargos  # 每个粒子的维度对应每个货物的起始位置

    # 如果未提供options，使用默认配置
    if options is None:
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # 定义适应度评估函数
    def fitness_function(x):
        """
        计算每个粒子的适应度值。

        参数:
            x (numpy.ndarray): 粒子的位置数组，形状为 (n_particles, dimension)。

        返回:
            numpy.ndarray: 每个粒子的适应度值。
        """
        fitness = np.zeros(x.shape[0])

        for idx, particle in enumerate(x):
            # 将连续位置映射为离散起始位置，根据货物类型对齐
            start_positions = []
            penalty = 0
            cg_change = 0.0
            occupied = np.zeros(num_positions, dtype=int)

            for i in range(num_cargos):
                cargo_type = cargo_types[i]
                pos_continuous = particle[i]

                # 根据货物类型对齐
                if cargo_type == 1:
                    start_pos = int(np.round(pos_continuous)) % num_positions
                elif cargo_type == 2:
                    start_pos = (int(np.round(pos_continuous)) // 2) * 2
                    if start_pos >= num_positions - 1:
                        start_pos = num_positions - 2
                elif cargo_type == 4:
                    start_pos = (int(np.round(pos_continuous)) // 4) * 4
                    if start_pos >= num_positions - 3:
                        start_pos = num_positions - 4
                else:
                    start_pos = 0  # 默认位置

                # 检查边界
                if start_pos < 0 or start_pos + cargo_type > num_positions:
                    penalty += 1000
                    continue

                # 检查对齐
                if cargo_type == 2 and start_pos % 2 != 0:
                    penalty += 1000
                if cargo_type == 4 and start_pos % 4 != 0:
                    penalty += 1000

                # 检查重叠
                if np.any(occupied[start_pos:start_pos + cargo_type]):
                    penalty += 1000
                else:
                    occupied[start_pos:start_pos + cargo_type] = 1

                start_positions.append(start_pos)

                # 计算重心变化量
                if cargo_type == 1:
                    cg_change += weights[i] * cg_impact[start_pos]
                elif cargo_type == 2:
                    cg_change += weights[i] * cg_impact_2u[start_pos // 2]
                elif cargo_type == 4:
                    cg_change += weights[i] * cg_impact_4u[start_pos // 4]

            fitness[idx] = cg_change + penalty

        return fitness

    # 设置PSO的边界
    # 对于每个货物，起始位置的范围根据货物类型对齐
    lower_bounds = []
    upper_bounds = []
    for i in range(num_cargos):
        cargo_type = cargo_types[i]
        if cargo_type == 1:
            lower_bounds.append(0)
            upper_bounds.append(num_positions - 1)
        elif cargo_type == 2:
            lower_bounds.append(0)
            upper_bounds.append(num_positions - 2)
        elif cargo_type == 4:
            lower_bounds.append(0)
            upper_bounds.append(num_positions - 4)
        else:
            lower_bounds.append(0)
            upper_bounds.append(num_positions - 1)

    bounds = (np.array(lower_bounds), np.array(upper_bounds))

    # 初始化PSO优化器
    optimizer = ps.single.GlobalBestPSO(n_particles=swarmsize, dimensions=dimension, options=options, bounds=bounds)

    # 运行PSO优化
    best_cost, best_pos = optimizer.optimize(fitness_function, iters=maxiter)

    # 将最佳位置映射为离散装载方案
    best_start_positions = []
    penalty = 0
    cg_change = 0.0
    occupied = np.zeros(num_positions, dtype=int)

    for i in range(num_cargos):
        cargo_type = cargo_types[i]
        pos_continuous = best_pos[i]

        # 根据货物类型对齐
        if cargo_type == 1:
            start_pos = int(np.round(pos_continuous)) % num_positions
        elif cargo_type == 2:
            start_pos = (int(np.round(pos_continuous)) // 2) * 2
            if start_pos >= num_positions - 1:
                start_pos = num_positions - 2
        elif cargo_type == 4:
            start_pos = (int(np.round(pos_continuous)) // 4) * 4
            if start_pos >= num_positions - 3:
                start_pos = num_positions - 4
        else:
            start_pos = 0  # 默认位置

        # 检查边界
        if start_pos < 0 or start_pos + cargo_type > num_positions:
            penalty += 1000
            best_start_positions.append(start_pos)
            continue

        # 检查对齐
        if cargo_type == 2 and start_pos % 2 != 0:
            penalty += 1000
        if cargo_type == 4 and start_pos % 4 != 0:
            penalty += 1000

        # 检查重叠
        if np.any(occupied[start_pos:start_pos + cargo_type]):
            penalty += 1000
        else:
            occupied[start_pos:start_pos + cargo_type] = 1

        best_start_positions.append(start_pos)

        # 计算重心变化量
        if cargo_type == 1:
            cg_change += abs(weights[i] * cg_impact[start_pos])
        elif cargo_type == 2:
            cg_change += abs(weights[i] * cg_impact_2u[start_pos // 2])
        elif cargo_type == 4:
            cg_change += abs(weights[i] * cg_impact_4u[start_pos // 4])

    total_cg_change = cg_change + penalty

    # 构建装载方案矩阵
    best_xij = np.zeros((num_cargos, num_positions), dtype=int)
    for i, start_pos in enumerate(best_start_positions):
        cargo_type = cargo_types[i]
        for k in range(cargo_type):
            pos = start_pos + k
            if pos < num_positions:
                best_xij[i, pos] = 1

    # 检查是否有严重惩罚，判断是否找到可行解
    if total_cg_change >= -999999:
        return best_xij, total_cg_change
    else:
        return [], -1000000

#
# # 示例输入和调用
def main():
    weights = [500, 800, 1200, 300, 700, 1000, 600, 900]  # 每个货物的质量
    cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']  # 货物名称
    cargo_types_dict = {"LD3": 1, "PLA": 2, "P6P": 4, "BULK": 1}  # 货物占位关系
    positions = list(range(44))  # 44个货位编号
    cg_impact = [i * 0.1 for i in range(44)]  # 每kg货物对重心index的影响系数 (单个位置)
    cg_impact_2u = [i * 0.08 for i in range(22)]  # 两个位置组合的影响系数
    cg_impact_4u = [i * 0.05 for i in range(11)]  # 四个位置组合的影响系数
    max_positions = 44  # 总货位数量

    # 设置PSO参数（可选）
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    solution, cg_change = cargo_load_planning_pso(
        weights, cargo_names, cargo_types_dict, positions,
        cg_impact, cg_impact_2u, cg_impact_4u, max_positions,
        options=options, swarmsize=200, maxiter=200
    )

    if solution is not None and len(solution) > 0:
        print("cargo assignment:")
        # print("装载方案矩阵:")
        # print(solution)
        # print(f"重心的变化量: {cg_change:.2f}")

        # 输出实际分布
        for i in range(len(weights)):
            assigned_positions = []
            for j in range(len(positions)):
                if solution[i, j] > 0.5:  # 判断位置是否被分配
                    assigned_positions.append(j)
            print(f"cargo {cargo_names[i]} -> {assigned_positions}")
    else:
        print("未找到可行的装载方案。")


if __name__ == "__main__":
    main()
