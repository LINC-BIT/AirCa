# import numpy as np
# from scipy.optimize import minimize
#
#
# def cargo_load_planning_nonlinear1(weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u,
#                         max_positions):
#     """
#     使用非线性优化方法计算货物装载方案，最小化重心的变化量。
#
#     参数:
#         weights (list): 每个货物的质量列表。
#         cargo_names (list): 每个货物的名称。
#         cargo_types_dict (dict): 货物名称和占用的货位数量。
#         positions (list): 可用的货位编号。
#         cg_impact (list): 每个位置每kg货物对重心index的影响系数。
#         cg_impact_2u (list): 两个位置组合的重心影响系数。
#         cg_impact_4u (list): 四个位置组合的重心影响系数。
#         max_positions (int): 总货位数量。
#
#     返回:
#         result.x: 最优装载方案矩阵。
#     """
#     # 将货物类型映射为对应的占用单位数
#     cargo_types = [cargo_types_dict[name] for name in cargo_names]
#
#     num_cargos = len(weights)  # 货物数量
#     num_positions = len(positions)  # 可用货位数量
#
#     # 决策变量：xij (是否将货物i放置在位置j)
#     def objective(x):
#         cg_change = 0
#         idx = 0
#         for i in range(num_cargos):
#             for j in range(num_positions):
#                 if cargo_types[i] == 1:
#                     cg_change += x[idx] * (weights[i] * cg_impact[j])**2
#                 elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
#                     cg_change += x[idx] * (weights[i] * cg_impact_2u[j // 2])**2
#                 elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
#                     cg_change += x[idx] * (weights[i] * cg_impact_4u[j // 4])**2
#                 idx += 1
#         return cg_change
#
#     # 约束：每个货物只能装载到一个位置
#     cons = []
#     for i in range(num_cargos):
#         cons.append({
#             'type': 'eq',  # 等式约束
#             'fun': lambda x, i=i: np.sum(x[i * num_positions:(i + 1) * num_positions]) - 1
#         })
#
#     # 约束：每个位置只能装载一个货物
#     for j in range(num_positions):
#         cons.append({
#             'type': 'eq',  # 等式约束
#             'fun': lambda x, j=j: np.sum(x[j::num_positions]) - 1
#         })
#
#     # 约束：占用多个位置的货物
#     for i, cargo_type in enumerate(cargo_types):
#         if cargo_type == 2:  # 两个连续位置组合
#             for j in range(0, num_positions - 1, 2):
#                 cons.append({
#                     'type': 'eq',
#                     'fun': lambda x, i=i, j=j: x[i * num_positions + j] + x[i * num_positions + j + 1] - 1
#                 })
#         elif cargo_type == 4:  # 四个连续位置组合
#             for j in range(0, num_positions - 3, 4):
#                 cons.append({
#                     'type': 'eq',
#                     'fun': lambda x, i=i, j=j: x[i * num_positions + j] + x[i * num_positions + j + 1] +
#                                                x[i * num_positions + j + 2] + x[i * num_positions + j + 3] - 1
#                 })
#
#     # 设置边界，变量只能是0或1
#     bounds = [(0, 1) for _ in range(num_cargos * num_positions)]
#
#     # 初始猜测 (假设每个货物都随机放置在一个位置)
#     x0 = np.zeros(num_cargos * num_positions)
#
#     # 使用SLSQP方法进行非线性优化求解
#     result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
#
#     if result.success:
#         solution = result.x.reshape((num_cargos, num_positions))
#
#         # 计算最终重心变化
#         cg_change = 0
#         idx = 0
#         for i in range(num_cargos):
#             for j in range(num_positions):
#                 if cargo_types[i] == 1:
#                     cg_change += solution[i, j] * (weights[i] * cg_impact[j])**2
#                 elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
#                     cg_change += solution[i, j] * (weights[i] * cg_impact_2u[j // 2])**2
#                 elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
#                     cg_change += solution[i, j] * (weights[i] * cg_impact_4u[j // 4])**2
#                 idx += 1
#
#         return result, cg_change
#     else:
#         return None, -1000000
#
#
# # 示例输入
# # def main():
# #     weights = [500, 800, 1200, 300, 700, 1000, 600, 900]  # 每个货物的质量
# #     cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']  # 货物名称
# #     cargo_types_dict = {"LD3": 1, "PLA": 2, "P6P": 4, "BULK": 1}  # 货物占位关系
# #     positions = list(range(44))  # 44个货位编号
# #     cg_impact = [i * 0.1 for i in range(44)]  # 每kg货物对重心index的影响系数 (单个位置)
# #     cg_impact_2u = [i * 0.08 for i in range(22)]  # 两个位置组合的影响系数
# #     cg_impact_4u = [i * 0.05 for i in range(11)]  # 四个位置组合的影响系数
# #     max_positions = 44  # 总货位数量
# #
# #     # 归一化 cg_impact 到 [-1, 1]
# #     cg_impact_min = min(cg_impact)
# #     cg_impact_max = max(cg_impact)
# #     cg_impact = [(2 * (x - cg_impact_min) / (cg_impact_max - cg_impact_min)) - 1 for x in cg_impact]
# #
# #     result, cg_change = cargo_load_planning_nonlinear1(weights, cargo_names, cargo_types_dict, positions, cg_impact,
# #                                                     cg_impact_2u, cg_impact_4u, max_positions)
# #
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# def main():
#     weights = [500, 800, 1200, 300, 700, 1000, 600, 900]  # 每个货物的质量
#     cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']  # 货物名称
#     cargo_types_dict = {"LD3": 1, "PLA": 2, "P6P": 4, "BULK": 1}  # 货物占位关系
#     positions = list(range(44))  # 44个货位编号
#     cg_impact = [i * 0.1 for i in range(44)]  # 每kg货物对重心index的影响系数 (单个位置)
#     cg_impact_2u = [i * 0.08 for i in range(22)]  # 两个位置组合的影响系数
#     cg_impact_4u = [i * 0.05 for i in range(11)]  # 四个位置组合的影响系数
#     max_positions = 44  # 总货位数量
#
#     # 归一化 cg_impact 到 [-1, 1]
#     cg_impact_min = min(cg_impact)
#     cg_impact_max = max(cg_impact)
#     cg_impact = [(2 * (x - cg_impact_min) / (cg_impact_max - cg_impact_min)) - 1 for x in cg_impact]
#
#     result, cg_change = cargo_load_planning_nonlinear1(weights, cargo_names, cargo_types_dict, positions, cg_impact,
#                                                     cg_impact_2u, cg_impact_4u, max_positions)
#
#     if result is not None:
#         solution = result.x.reshape((len(weights), len(positions)))
#
#         print("\n最优货物摆放方案：")
#         for i in range(len(weights)):
#             assigned_positions = [positions[j] for j in range(len(positions)) if solution[i, j] > 0.5]  # 找到货物的货位
#             print(f"货物 {cargo_names[i]} (重量: {weights[i]}kg) → 位置: {assigned_positions}")
#
#         print(f"\n最终重心变化: {cg_change:.2f}")
#     else:
#         print("优化失败，未找到可行的货物摆放方案。")
#
#
# if __name__ == "__main__":
#     main()
#
import numpy as np
from scipy.optimize import minimize


def cargo_load_planning_nonlinear1(weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u,
                                   max_positions):
    """
    使用非线性优化方法计算货物装载方案，最小化重心的变化量。

    参数:
        weights (list): 每个货物的质量列表。
        cargo_names (list): 每个货物的名称。
        cargo_types_dict (dict): 货物名称和占用的货位数量。
        positions (list): 可用的货位编号。
        cg_impact (list): 每个位置每kg货物对重心index的影响系数(单个位置)。
        cg_impact_2u (list): 两个位置组合的重心影响系数(这里演示可以空着)。
        cg_impact_4u (list): 四个位置组合的重心影响系数(这里演示可以空着)。
        max_positions (int): 总货位数量。

    返回:
        (result, cg_change):
            result 是 scipy.optimize.minimize 的优化结果对象，
            cg_change 是按照解算出来的摆放方案计算的目标函数值(重心变化量)。
    """
    # 1) 货物类型映射为所占的单位数
    cargo_types = [cargo_types_dict[name] for name in cargo_names]

    num_cargos = len(weights)     # 货物数量
    num_positions = len(positions)  # 可用货位数量

    # 2) 目标函数
    def objective(x):
        cg_change = 0
        idx = 0
        for i in range(num_cargos):
            for j in range(num_positions):
                # 判断货物类型
                if cargo_types[i] == 1:
                    cg_change += x[idx] * (weights[i] * cg_impact[j]) ** 2
                elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
                    cg_change += x[idx] * (weights[i] * cg_impact_2u[j // 2]) ** 2
                elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
                    cg_change += x[idx] * (weights[i] * cg_impact_4u[j // 4]) ** 2
                idx += 1
        return cg_change

    # 3) 约束
    cons = []
    # (a) 每个货物恰好装载到一个位置: sum_j x[i,j] = 1
    for i in range(num_cargos):
        cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: np.sum(x[i * num_positions:(i + 1) * num_positions]) - 1
        })

    # (b) 每个位置只能装载一个货物: sum_i x[i,j] = 1
    # 如果你想允许空位，可改成 ≤ 1，而不是 = 1
    for j in range(num_positions):
        cons.append({
            'type': 'eq',
            'fun': lambda x, j=j: np.sum(x[j::num_positions]) - 1
        })

    # (c) 多位置占用的货物(如果全部货物都是 type=1，就不会生效)
    for i, cargo_type in enumerate(cargo_types):
        if cargo_type == 2:  # 两个连续位置组合
            for j in range(0, num_positions - 1, 2):
                cons.append({
                    'type': 'eq',
                    'fun': lambda x, i=i, j=j: x[i * num_positions + j] + x[i * num_positions + j + 1] - 1
                })
        elif cargo_type == 4:  # 四个连续位置组合
            for j in range(0, num_positions - 3, 4):
                cons.append({
                    'type': 'eq',
                    'fun': lambda x, i=i, j=j: x[i * num_positions + j] + x[i * num_positions + j + 1] +
                                               x[i * num_positions + j + 2] + x[i * num_positions + j + 3] - 1
                })

    # 4) 变量边界 0 ≤ x ≤ 1
    bounds = [(0, 1) for _ in range(num_cargos * num_positions)]

    # 5) 初始猜测
    x0 = np.zeros(num_cargos * num_positions)

    # 6) 调用 SLSQP 优化
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

    # 7) 如果优化成功，就计算最终目标值
    if result.success:
        solution = result.x.reshape((num_cargos, num_positions))

        cg_change = 0
        idx = 0
        for i in range(num_cargos):
            for j in range(num_positions):
                if cargo_types[i] == 1:
                    cg_change += solution[i, j] * (weights[i] * cg_impact[j]) ** 2
                elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
                    cg_change += solution[i, j] * (weights[i] * cg_impact_2u[j // 2]) ** 2
                elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
                    cg_change += solution[i, j] * (weights[i] * cg_impact_4u[j // 4]) ** 2
                idx += 1

        return result, cg_change
    else:
        return None, -1e10


def main():
    """
    这里给出一个“最简可行的示例”：
    - 有 4 个货物，全部类型为 1（每个货物只占一个位置）。
    - 有 4 个可用位置。
    - 强制每个位置都放一个货物（如果想留空，可以把位置约束改为 <=1）。
    """

    # 1) 定义货物及其类型
    weights = [500, 800, 300, 700]  # 4 个货物的质量
    cargo_names = ["LD3", "LD3", "LD3", "LD3"]  # 全部是同一种类型
    cargo_types_dict = {"LD3": 1}  # 所有 LD3 都占 1 个位置

    # 2) 定义货位(4 个)
    positions = [0, 1, 2, 3]  # 仅 4 个位置
    max_positions = len(positions)

    # 3) 设定重心影响因子(随便给，越往后位置影响系数越大)
    cg_impact = [0.1, 0.2, 0.3, 0.4]

    # 这两个列表先空着，因为此示例不需要 2 位 或 4 位组合
    cg_impact_2u = []
    cg_impact_4u = []

    # 4) 调用非线性优化
    result, cg_change = cargo_load_planning_nonlinear1(weights, cargo_names, cargo_types_dict,
                                                       positions, cg_impact,
                                                       cg_impact_2u, cg_impact_4u,
                                                       max_positions)

    if result is not None:
        solution = result.x.reshape((len(weights), len(positions)))

        # 打印结果
        print("cargo assignment:")
        for i in range(len(weights)):
            # 找到货物 i 的货位
            # 注意：因为是 0/1 变量，通常可以判断 > 0.5 就算 1
            assigned_positions = [positions[j] for j in range(len(positions)) if solution[i, j] > 0.5]
            print(f"cargo {cargo_names[i]} → {assigned_positions}")

        # print(f"\n最终重心变化(目标函数值): {cg_change:.4f}")
    else:
        print("优化失败，未找到可行解。")


if __name__ == "__main__":
    main()
