#%%
import numpy as np
from scipy.optimize import linprog


def cargo_load_planning_linear2(weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u,
                        max_positions):
    """
    使用整数线性规划方法计算货物装载方案，最小化重心的变化量。

    参数:
        weights (list): 每个货物的质量列表。
        cargo_names (list): 每个货物的名称。
        cargo_types_dict (dict): 货物名称和占用的货位数量。
        positions (list): 可用的货位编号。
        cg_impact (list): 每个位置每kg货物对重心index的影响系数。
        cg_impact_2u (list): 两个位置组合的重心影响系数。
        cg_impact_4u (list): 四个位置组合的重心影响系数。
        max_positions (int): 总货位的数量。

    返回:
        result.x: 最优装载方案矩阵。
    """
    # 将货物类型映射为对应的占用单位数
    cargo_types = [cargo_types_dict[name] for name in cargo_names]

    num_cargos = len(weights)  # 货物数量
    num_positions = len(positions)  # 可用货位数量

    # 决策变量：xij (是否将货物i放置在位置j)
    c = []  # 目标函数系数列表
    for i in range(num_cargos):
        for j in range(num_positions):
            if cargo_types[i] == 1:
                c.append(abs(weights[i] * cg_impact[j]))
            elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
                c.append(abs(weights[i] * cg_impact_2u[j // 2]))
            elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
                c.append(abs(weights[i] * cg_impact_4u[j // 4]))
            else:
                c.append(0)  # 不适合的索引默认影响为0

    # 决策变量约束：xij只能是0或1 (整型约束由 linprog 近似处理)
    bounds = [(0, 1) for _ in range(num_cargos * num_positions)]

    # 约束1：每个货物只能装载到一个位置
    A_eq = []
    b_eq = []
    for i in range(num_cargos):
        constraint = [0] * (num_cargos * num_positions)
        for j in range(num_positions):
            constraint[i * num_positions + j] = 1
        A_eq.append(constraint)
        b_eq.append(1)

    # 约束2：每个位置只能装载一个货物
    A_ub = []
    b_ub = []
    for j in range(num_positions):  # 遍历所有位置
        constraint = [0] * (num_cargos * num_positions)
        for i in range(num_cargos):  # 遍历所有货物
            constraint[i * num_positions + j] = 1
        A_ub.append(constraint)
        b_ub.append(1)  # 每个位置最多只能分配一个货物

    # 约束3：占用多个位置的货物
    for i, cargo_type in enumerate(cargo_types):
        if cargo_type == 2:  # 两个连续位置组合
            for j in range(0, num_positions - 1, 2):
                constraint = [0] * (num_cargos * num_positions)
                constraint[i * num_positions + j] = 1
                constraint[i * num_positions + j + 1] = 1
                A_ub.append(constraint)
                b_ub.append(1)
        elif cargo_type == 4:  # 上两个、下两个组合
            for j in range(0, num_positions - 3, 4):
                constraint = [0] * (num_cargos * num_positions)
                constraint[i * num_positions + j] = 1
                constraint[i * num_positions + j + 1] = 1
                constraint[i * num_positions + j + 2] = 1
                constraint[i * num_positions + j + 3] = 1
                A_ub.append(constraint)
                b_ub.append(1)

    # 转换为numpy数组
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array(c)

    # 求解线性规划问题
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')

    if result.success:
        # 解决方案
        solution = result.x.reshape((num_cargos, num_positions))

        # 计算最终重心变化
        cg_change = 0
        for i in range(num_cargos):
            for j in range(num_positions):
                if cargo_types[i] == 1:
                    cg_change += solution[i, j] * weights[i] * cg_impact[j]
                elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
                    cg_change += solution[i, j] * weights[i] * cg_impact_2u[j // 2]
                elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
                    cg_change += solution[i, j] * weights[i] * cg_impact_4u[j // 4]


        # 输出货物实际分布
        print("cargo assignment:")
        for i in range(num_cargos):
            assigned_positions = []
            for j in range(num_positions):
                if solution[i, j] > 0.5:  # 判断位置是否被分配
                    assigned_positions.append(j)
            print(f"cargo {cargo_names[i]} -> {assigned_positions}")

        # 输出1号LD3的摆放位置
        ld3_positions = []
        for j in range(num_positions):
            if solution[0, j] > 0.5:  # LD3 是第一个货物
                ld3_positions.append(j)

        # print(f"1号LD3的摆放位置: {ld3_positions}")
        return result, cg_change

    else:
        return None, -1000000


# 示例输入
def main():
    weights = [500, 800, 1200, 300, 700, 1000, 600, 900]  # 每个货物的质量
    cargo_names = ['PLA', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']  # 货物名称
    cargo_types_dict = {"LD3": 1, "PLA": 2, "P6P": 4, "BULK": 1}  # 货物占位关系
    positions = list(range(44))  # 44个货位编号
    cg_impact = [i * 0.1 for i in range(44)]  # 每kg货物对重心index的影响系数 (单个位置)
    cg_impact_2u = [i * 0.08 for i in range(22)]  # 两个位置组合的影响系数
    cg_impact_4u = [i * 0.05 for i in range(11)]  # 四个位置组合的影响系数
    max_positions = 44  # 总货位数量

    # 归一化 cg_impact 到 [-1, 1]
    cg_impact_min = min(cg_impact)
    cg_impact_max = max(cg_impact)
    cg_impact = [(2 * (x - cg_impact_min) / (cg_impact_max - cg_impact_min)) - 1 for x in cg_impact]

    result, cg_change = cargo_load_planning_linear2(weights, cargo_names, cargo_types_dict, positions, cg_impact,
                                                    cg_impact_2u, cg_impact_4u, max_positions)
    # print(f"最小重心变化量: {cg_change:.2f}")


if __name__ == "__main__":
    main()
