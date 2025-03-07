
import numpy as np


def cargo_load_planning_heuristic(weights, cargo_names, cargo_types_dict, positions,
                                  cg_impact, cg_impact_2u, cg_impact_4u, max_positions,
                                  max_iterations=1000):
    """
    使用两阶段启发式算法计算货物装载方案，最小化重心的变化量。

    参数:
        weights (list): 每个货物的质量列表。
        cargo_names (list): 每个货物的名称。
        cargo_types_dict (dict): 货物名称和占用的货位数量。
        positions (list): 可用的货位编号。
        cg_impact (list): 每个位置每kg货物对重心index的影响系数。
        cg_impact_2u (list): 两个位置组合的重心影响系数。
        cg_impact_4u (list): 四个位置组合的重心影响系数。
        max_positions (int): 总货位的数量。
        max_iterations (int): 优化阶段的最大迭代次数。

    返回:
        solution (np.array): 最优装载方案矩阵。
        total_cg_change (float): 最优方案的重心变化量。
    """
    # 将货物类型映射为对应的占用单位数
    cargo_types = [cargo_types_dict[name] for name in cargo_names]

    num_cargos = len(weights)  # 货物数量
    num_positions = len(positions)  # 可用货位数量

    # 初始化装载方案矩阵
    solution = np.zeros((num_cargos, num_positions), dtype=int)

    # 标记已占用的位置
    occupied = np.zeros(num_positions, dtype=int)

    # 按照货物类型（占用单位数降序）和重量降序排序货物
    sorted_indices = sorted(range(num_cargos), key=lambda i: (-cargo_types[i], -weights[i]))

    # 阶段1：初始装载方案生成
    for i in sorted_indices:
        cargo_type = cargo_types[i]
        feasible_positions = []

        # 根据货物类型确定可行的起始位置
        if cargo_type == 1:
            possible_starts = range(0, num_positions)
        elif cargo_type == 2:
            possible_starts = range(0, num_positions - 1, 2)
        elif cargo_type == 4:
            possible_starts = range(0, num_positions - 3, 4)
        else:
            possible_starts = []

        for start in possible_starts:
            # 检查是否超出边界
            if start + cargo_type > num_positions:
                continue
            # 检查是否与已占用位置重叠
            if np.any(occupied[start:start + cargo_type]):
                continue
            # 计算重心变化量
            if cargo_type == 1:
                cg = abs(weights[i] * cg_impact[start])
            elif cargo_type == 2:
                cg = abs(weights[i] * cg_impact_2u[start // 2])
            elif cargo_type == 4:
                cg = abs(weights[i] * cg_impact_4u[start // 4])
            feasible_positions.append((start, cg))

        # 如果有可行位置，选择使重心变化最小的位置
        if feasible_positions:
            best_start, best_cg = min(feasible_positions, key=lambda x: x[1])
            solution[i, best_start:best_start + cargo_type] = 1
            occupied[best_start:best_start + cargo_type] = 1
        else:
            # 如果没有可行位置，则尝试分配到任何未占用的位置（可能违反约束）
            for start in range(0, num_positions - cargo_type + 1):
                if np.all(occupied[start:start + cargo_type] == 0):
                    solution[i, start:start + cargo_type] = 1
                    occupied[start:start + cargo_type] = 1
                    break

    # 计算初始重心变化量
    total_cg_change = 0.0
    for i in range(num_cargos):
        cargo_type = cargo_types[i]
        assigned_positions = np.where(solution[i] == 1)[0]
        if len(assigned_positions) == 0:
            continue
        if cargo_type == 1:
            total_cg_change += abs(weights[i] * cg_impact[assigned_positions[0]])
        elif cargo_type == 2:
            total_cg_change += abs(weights[i] * cg_impact_2u[assigned_positions[0] // 2])
        elif cargo_type == 4:
            total_cg_change += abs(weights[i] * cg_impact_4u[assigned_positions[0] // 4])

    # 阶段2：装载方案优化
    for _ in range(max_iterations):
        improved = False
        for i in range(num_cargos):
            cargo_type = cargo_types[i]
            current_positions = np.where(solution[i] == 1)[0]
            if len(current_positions) == 0:
                continue
            current_start = current_positions[0]

            # 根据货物类型确定可行的起始位置
            if cargo_type == 1:
                possible_starts = range(0, num_positions)
            elif cargo_type == 2:
                possible_starts = range(0, num_positions - 1, 2)
            elif cargo_type == 4:
                possible_starts = range(0, num_positions - 3, 4)
            else:
                possible_starts = []

            best_start = current_start
            best_cg = total_cg_change

            for start in possible_starts:
                if start == current_start:
                    continue
                # 检查是否超出边界
                if start + cargo_type > num_positions:
                    continue
                # 检查是否与已占用位置重叠
                if np.any(occupied[start:start + cargo_type]):
                    continue
                # 计算重心变化量
                if cargo_type == 1:
                    new_cg = abs(weights[i] * cg_impact[start])
                elif cargo_type == 2:
                    new_cg = abs(weights[i] * cg_impact_2u[start // 2])
                elif cargo_type == 4:
                    new_cg = abs(weights[i] * cg_impact_4u[start // 4])
                else:
                    new_cg = 0

                # 计算新的总重心变化量
                temp_cg_change = total_cg_change - (
                    weights[i] * cg_impact[current_start] if cargo_type == 1 else
                    weights[i] * cg_impact_2u[current_start // 2] if cargo_type == 2 else
                    weights[i] * cg_impact_4u[current_start // 4] if cargo_type == 4 else 0
                ) + new_cg

                # 如果新的重心变化量更小，进行更新
                if temp_cg_change < best_cg:
                    best_cg = temp_cg_change
                    best_start = start

            # 如果找到了更好的位置，进行更新
            if best_start != current_start:
                # 释放当前占用的位置
                occupied[current_start:current_start + cargo_type] = 0
                solution[i, current_start:current_start + cargo_type] = 0

                # 分配到新的位置
                solution[i, best_start:best_start + cargo_type] = 1
                occupied[best_start:best_start + cargo_type] = 1

                # 更新总重心变化量
                total_cg_change = best_cg
                improved = True

        if not improved:
            break  # 如果在一个完整的迭代中没有改进，结束优化

    return solution, total_cg_change


# 示例输入和调用
def main():
    weights = [500, 800, 1200, 300, 700, 1000, 600, 900]  # 每个货物的质量
    cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']  # 货物名称
    cargo_types_dict = {"LD3": 1, "PLA": 2, "P6P": 4, "BULK": 1}  # 货物占位关系
    positions = list(range(44))  # 44个货位编号
    cg_impact = [i * 0.1 for i in range(44)]  # 每kg货物对重心index的影响系数 (单个位置)
    cg_impact_2u = [i * 0.08 for i in range(22)]  # 两个位置组合的影响系数
    cg_impact_4u = [i * 0.05 for i in range(11)]  # 四个位置组合的影响系数
    max_positions = 44  # 总货位数量

    solution, cg_change = cargo_load_planning_heuristic(
        weights, cargo_names, cargo_types_dict, positions,
        cg_impact, cg_impact_2u, cg_impact_4u, max_positions,
        max_iterations=1000
    )

    if solution is not None and solution.size > 0:
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

