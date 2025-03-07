import numpy as np
import pulp

def cargo_load_planning_mip(weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u,
                            max_positions):
    """
    使用混合整数规划方法计算货物装载方案，最小化重心的变化量。

    参数:
        weights (list): 每个货物的质量列表。
        cargo_names (list): 每个货物的名称。
        cargo_types_dict (dict): 货物名称和占用的货位数量。
        positions (list): 可用的货位编号。
        cg_impact (list): 每个位置每kg货物对重心index的影响系数。
        cg_impact_2u (list): 两个位置组合的重心影响系数。
        cg_impact_4u (list): 四个位置组合的重心影响系数。
        max_positions (int): 总货位数量。

    返回:
        result (pulp.LpStatus): 求解状态。
        cg_change (float): 最优解的重心变化量。
        solution_matrix (np.ndarray): 最优装载方案矩阵。
    """
    # 将货物类型映射为对应的占用单位数
    cargo_types = [cargo_types_dict[name] for name in cargo_names]

    num_cargos = len(weights)        # 货物数量
    num_positions = len(positions)   # 可用货位数量

    # 创建优化问题实例
    prob = pulp.LpProblem("Cargo_Load_Planning", pulp.LpMinimize)

    # 创建决策变量 x_ij (是否将货物i放置在位置j)
    # 使用字典键 (i,j) 来标识变量
    x = pulp.LpVariable.dicts("x",
                              ((i, j) for i in range(num_cargos) for j in range(num_positions)),
                              cat='Binary')

    # 定义目标函数：最小化重心的变化量
    objective_terms = []
    for i in range(num_cargos):
        for j in range(num_positions):
            if cargo_types[i] == 1:
                impact = abs(weights[i] * cg_impact[j])
            elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
                impact = abs(weights[i] * cg_impact_2u[j // 2])
            elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
                impact = abs(weights[i] * cg_impact_4u[j // 4])
            else:
                impact = 0
            objective_terms.append(impact * x[i, j])

    prob += pulp.lpSum(objective_terms), "Total_CG_Change"

    # 约束1：每个货物只能装载到一个位置
    for i in range(num_cargos):
        prob += pulp.lpSum([x[i, j] for j in range(num_positions)]) == 1, f"Cargo_{i}_Single_Position"

    # 约束2：每个位置只能装载一个货物
    for j in range(num_positions):
        prob += pulp.lpSum([x[i, j] for i in range(num_cargos)]) <= 1, f"Position_{j}_Single_Cargo"

    # 约束3：占用多个位置的货物
    for i, cargo_type in enumerate(cargo_types):
        if cargo_type == 2:  # 两个连续位置组合
            for j in range(0, num_positions - 1, 2):
                prob += x[i, j] + x[i, j + 1] <= 1, f"Cargo_{i}_Type2_Position_{j}_{j+1}"
        elif cargo_type == 4:  # 四个连续位置组合
            for j in range(0, num_positions - 3, 4):
                prob += x[i, j] + x[i, j + 1] + x[i, j + 2] + x[i, j + 3] <= 1, f"Cargo_{i}_Type4_Position_{j}_{j+3}"

    # 求解问题
    solver = pulp.PULP_CBC_CMD(msg=False)  # 使用默认的CBC求解器，不显示求解过程
    prob.solve(solver)

    # 检查求解状态
    if pulp.LpStatus[prob.status] == 'Optimal':
        # 构建装载方案矩阵
        solution = np.zeros((num_cargos, num_positions))
        for i in range(num_cargos):
            for j in range(num_positions):
                var_value = pulp.value(x[i, j])
                if var_value is not None and var_value > 0.5:
                    solution[i, j] = 1

        # 计算最终重心变化
        cg_change = 0.0
        for i in range(num_cargos):
            for j in range(num_positions):
                if solution[i, j] == 1:
                    if cargo_types[i] == 1:
                        cg_change += weights[i] * cg_impact[j]
                    elif cargo_types[i] == 2 and j % 2 == 0 and j < len(cg_impact_2u) * 2:
                        cg_change += weights[i] * cg_impact_2u[j // 2]
                    elif cargo_types[i] == 4 and j % 4 == 0 and j < len(cg_impact_4u) * 4:
                        cg_change += weights[i] * cg_impact_4u[j // 4]

        return solution, cg_change
    else:
        # 若求解失败，则返回空结果和错误标志
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

    # 归一化 cg_impact 到 [-1, 1]
    cg_impact_min = min(cg_impact)
    cg_impact_max = max(cg_impact)
    cg_impact = [(2 * (x - cg_impact_min) / (cg_impact_max - cg_impact_min)) - 1 for x in cg_impact]

    # 调用装载规划函数
    solution, cg_change = cargo_load_planning_mip(
        weights, cargo_names, cargo_types_dict, positions, cg_impact, cg_impact_2u, cg_impact_4u, max_positions
    )
    # print(solution)
    if cg_change!=-1000000:
        print("cargo assignment")
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
        print("未能找到可行解。")
        print(f"求解状态: {pulp.LpStatus[solution]}")

if __name__ == "__main__":
    main()


