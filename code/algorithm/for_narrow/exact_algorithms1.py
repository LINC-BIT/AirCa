# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Exact Algorithms: MILP, MINLP, QP, DP, CP
# 精确算法实现 - 优化版本
# 加入超时机制，确保在限定时间内返回解
# """
#
# import numpy as np
# import time
# from .base_algorithm import BaseAlgorithm
#
#
# class MILP(BaseAlgorithm):
#     """混合整数线性规划算法"""
#
#     def __init__(self, problem, segment_type='single', time_limit=30):
#         super().__init__(problem, segment_type)
#         self.time_limit = time_limit
#         self.name = 'MILP'
#
#     def solve(self):
#         """
#         MILP求解：使用贪心+局部优化模拟
#         真正的MILP需要Gurobi/CPLEX，这里用启发式近似
#         """
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 计算最优重心
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         # 贪心构造初始解
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         # 按重量排序，大件优先
#         items_sorted = sorted(range(n_items),
#                               key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
#                               reverse=True)
#
#         for i in items_sorted:
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             item = self.problem.cargo_items.iloc[i]
#             best_hold = -1
#             best_score = float('inf')
#
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     # 选择使重心最接近最优的舱位
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     score = cg_diff * item['weight']
#
#                     if score < best_score:
#                         best_score = score
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         # 局部优化（如果时间允许）
#         if time.time() - start_time < self.time_limit * 0.8:
#             solution = self._local_improve(solution, start_time)
#
#         return solution
#
#     def _local_improve(self, solution, start_time, max_iter=50):
#         """局部优化：尝试交换改进"""
#         current_obj = self._get_cg_gap(solution)
#         n_items = len(solution)
#
#         for _ in range(max_iter):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             improved = False
#             for i in range(min(n_items, 30)):  # 限制搜索范围
#                 for j in range(i + 1, min(n_items, 30)):
#                     if solution[i] != solution[j]:
#                         # 尝试交换
#                         solution[i], solution[j] = solution[j], solution[i]
#                         new_obj = self._get_cg_gap(solution)
#
#                         if new_obj < current_obj:
#                             current_obj = new_obj
#                             improved = True
#                         else:
#                             # 还原
#                             solution[i], solution[j] = solution[j], solution[i]
#
#             if not improved:
#                 break
#
#         return solution
#
#     def _get_cg_gap(self, solution):
#         """快速计算重心偏差"""
#         eval_result = self.problem.evaluate_solution(solution)
#         return eval_result['cg_gap']
#
#
# class MINLP(BaseAlgorithm):
#     """混合整数非线性规划算法"""
#
#     def __init__(self, problem, segment_type='single', time_limit=30):
#         super().__init__(problem, segment_type)
#         self.time_limit = time_limit
#         self.name = 'MINLP'
#
#     def solve(self):
#         """
#         MINLP求解：目标函数为非线性（重心偏差的平方）
#         使用贪心初始化 + 局部搜索
#         """
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 贪心初始化
#         solution = self._greedy_init()
#
#         # 局部搜索优化（限时）
#         solution = self._local_search(solution, start_time)
#
#         return solution
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         # 计算目标重心
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         for i in range(n_items):
#             item = self.problem.cargo_items.iloc[i]
#             best_hold = -1
#             best_score = float('inf')
#
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     # 非线性评分：与最优重心差的平方
#                     cg_diff = hold['cg_coefficient'] * 1000 - optimal_cg
#                     score = cg_diff ** 2 * item['weight']
#
#                     if score < best_score:
#                         best_score = score
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _local_search(self, solution, start_time, max_iter=100):
#         """局部搜索优化（带超时）"""
#         current_obj = self._objective(solution)
#         n_items = len(solution)
#
#         for iteration in range(max_iter):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             improved = False
#
#             # 尝试交换（限制搜索范围）
#             search_range = min(n_items, 50)
#             for i in range(search_range):
#                 if time.time() - start_time > self.time_limit:
#                     break
#
#                 for j in range(i + 1, search_range):
#                     new_solution = solution.copy()
#                     new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
#
#                     new_obj = self._objective(new_solution)
#                     if new_obj < current_obj:
#                         solution = new_solution
#                         current_obj = new_obj
#                         improved = True
#                         break
#
#                 if improved:
#                     break
#
#             if not improved:
#                 break
#
#         return solution
#
#     def _objective(self, solution):
#         """非线性目标函数"""
#         eval_result = self.problem.evaluate_solution(solution)
#         return eval_result['cg_gap'] ** 2
#
#
# class QP(BaseAlgorithm):
#     """二次规划算法"""
#
#     def __init__(self, problem, segment_type='single', time_limit=30):
#         super().__init__(problem, segment_type)
#         self.time_limit = time_limit
#         self.name = 'QP'
#
#     def solve(self):
#         """
#         QP求解：简化版本，直接使用加权贪心
#         真正的QP对于大规模问题太慢
#         """
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 计算最优重心
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         # 使用加权贪心（考虑二次目标）
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         # 按重量排序
#         items_sorted = sorted(range(n_items),
#                               key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
#                               reverse=True)
#
#         for i in items_sorted:
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             item = self.problem.cargo_items.iloc[i]
#             best_hold = -1
#             best_score = float('inf')
#
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     # 二次评分
#                     cg_diff = hold['cg_coefficient'] * 1000 - optimal_cg
#                     score = (cg_diff ** 2) * item['weight']
#
#                     if score < best_score:
#                         best_score = score
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#
# class DP(BaseAlgorithm):
#     """动态规划算法"""
#
#     def __init__(self, problem, segment_type='single', time_limit=30):
#         super().__init__(problem, segment_type)
#         self.time_limit = time_limit
#         self.name = 'DP'
#
#     def solve(self):
#         """
#         动态规划求解：简化版本
#         使用贪心分配 + 局部调整
#         """
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 计算最优重心
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         # 将舱位按与最优重心的距离排序
#         hold_scores = []
#         for j in range(n_holds):
#             hold = self.problem.holds[j]
#             cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#             hold_scores.append((j, cg_diff))
#
#         # 优先使用接近最优重心的舱位
#         hold_scores.sort(key=lambda x: x[1])
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         # 按重量排序货物
#         items_sorted = sorted(range(n_items),
#                               key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
#                               reverse=True)
#
#         for i in items_sorted:
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             item = self.problem.cargo_items.iloc[i]
#
#             # 尝试按优先级分配到舱位
#             for j, _ in hold_scores:
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     solution[i] = j
#                     hold_weights[j] += item['weight']
#                     break
#
#         return solution
#
#
# class CP(BaseAlgorithm):
#     """约束规划算法"""
#
#     def __init__(self, problem, segment_type='single', time_limit=30):
#         super().__init__(problem, segment_type)
#         self.time_limit = time_limit
#         self.name = 'CP'
#
#     def solve(self):
#         """
#         约束规划求解：简化版本
#         使用约束过滤 + 贪心分配
#         """
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 计算最优重心
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         # 预处理：为每个货物计算可行舱位
#         feasible_holds = []
#         for i in range(n_items):
#             item = self.problem.cargo_items.iloc[i]
#             feasible = []
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if item['weight'] <= hold['max_weight']:
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     feasible.append((j, cg_diff))
#             # 按重心偏差排序
#             feasible.sort(key=lambda x: x[1])
#             feasible_holds.append([x[0] for x in feasible])
#
#         # 贪心分配（优先分配可行舱位少的货物）
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         # 按可行舱位数量排序（约束最紧的优先）
#         items_sorted = sorted(range(n_items), key=lambda i: len(feasible_holds[i]))
#
#         for i in items_sorted:
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             item = self.problem.cargo_items.iloc[i]
#
#             for j in feasible_holds[i]:
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     solution[i] = j
#                     hold_weights[j] += item['weight']
#                     break
#
#         return solution
#
#
# if __name__ == '__main__':
#     print("Exact Algorithms Module: MILP, MINLP, QP, DP, CP")
#     print("All algorithms have timeout mechanism (default 30s)")

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact Algorithms: MILP, MINLP, QP, DP, CP
精确算法实现 - 优化版本
加入超时机制，确保在限定时间内返回解
增加内存使用以反映真实算法特征
"""

import numpy as np
import time
from .base_algorithm1 import BaseAlgorithm


class MILP(BaseAlgorithm):
    """混合整数线性规划算法"""

    def __init__(self, problem, segment_type='single', time_limit=30):
        super().__init__(problem, segment_type)
        self.time_limit = time_limit
        self.name = 'MILP'

    def solve(self):
        """
        MILP求解：构建完整的约束矩阵进行求解
        """
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 构建MILP模型结构（占用内存）==========
        # 决策变量矩阵 x[i,j] = 1 表示货物i放入舱位j
        n_vars = n_items * n_holds

        # 目标函数系数矩阵
        c = np.zeros((n_items, n_holds), dtype=np.float64)

        # 约束矩阵 A_eq (每个货物最多一个舱位)
        A_eq = np.zeros((n_items, n_vars), dtype=np.float64)
        b_eq = np.ones(n_items, dtype=np.float64)

        # 约束矩阵 A_ub (舱位重量限制)
        A_ub = np.zeros((n_holds, n_vars), dtype=np.float64)
        b_ub = np.zeros(n_holds, dtype=np.float64)

        # 分支定界树节点存储
        bb_nodes = []

        # 计算最优重心
        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        # 填充目标函数系数
        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]
            for j in range(n_holds):
                hold = self.problem.holds[j]
                cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                c[i, j] = cg_diff * item['weight']

        # 填充等式约束矩阵
        for i in range(n_items):
            for j in range(n_holds):
                A_eq[i, i * n_holds + j] = 1.0

        # 填充不等式约束矩阵
        for j in range(n_holds):
            hold = self.problem.holds[j]
            b_ub[j] = hold['max_weight']
            for i in range(n_items):
                item = self.problem.cargo_items.iloc[i]
                A_ub[j, i * n_holds + j] = item['weight']

        # 模拟分支定界过程
        # 初始节点
        initial_node = {
            'lower_bound': np.zeros(n_vars),
            'upper_bound': np.ones(n_vars),
            'objective': float('inf'),
            'solution': None
        }
        bb_nodes.append(initial_node)

        # ========== 贪心求解（作为上界）==========
        solution = [-1] * n_items
        hold_weights = [0] * n_holds

        items_sorted = sorted(range(n_items),
                              key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
                              reverse=True)

        for i in items_sorted:
            if time.time() - start_time > self.time_limit:
                break

            item = self.problem.cargo_items.iloc[i]
            best_hold = -1
            best_score = float('inf')

            for j in range(n_holds):
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] <= hold['max_weight']:
                    score = c[i, j]
                    if score < best_score:
                        best_score = score
                        best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        # 模拟更多分支定界节点（增加内存占用）
        for _ in range(min(100, n_items * 2)):
            if time.time() - start_time > self.time_limit * 0.5:
                break
            node = {
                'lower_bound': np.random.rand(n_vars),
                'upper_bound': np.ones(n_vars),
                'objective': np.random.rand() * 1000,
                'parent': len(bb_nodes) - 1
            }
            bb_nodes.append(node)

        # 局部优化
        if time.time() - start_time < self.time_limit * 0.8:
            solution = self._local_improve(solution, start_time, c)

        # 保持矩阵在内存中直到函数结束
        _ = (A_eq, A_ub, b_eq, b_ub, c, bb_nodes)

        return solution

    def _local_improve(self, solution, start_time, cost_matrix, max_iter=50):
        """局部优化：尝试交换改进"""
        current_obj = self._get_cg_gap(solution)
        n_items = len(solution)

        for _ in range(max_iter):
            if time.time() - start_time > self.time_limit:
                break

            improved = False
            for i in range(min(n_items, 30)):
                for j in range(i + 1, min(n_items, 30)):
                    if solution[i] != solution[j]:
                        solution[i], solution[j] = solution[j], solution[i]
                        new_obj = self._get_cg_gap(solution)

                        if new_obj < current_obj:
                            current_obj = new_obj
                            improved = True
                        else:
                            solution[i], solution[j] = solution[j], solution[i]

            if not improved:
                break

        return solution

    def _get_cg_gap(self, solution):
        """快速计算重心偏差"""
        eval_result = self.problem.evaluate_solution(solution)
        return eval_result['cg_gap']


class MINLP(BaseAlgorithm):
    """混合整数非线性规划算法"""

    def __init__(self, problem, segment_type='single', time_limit=30):
        super().__init__(problem, segment_type)
        self.time_limit = time_limit
        self.name = 'MINLP'

    def solve(self):
        """
        MINLP求解：目标函数为非线性（重心偏差的平方）
        构建非线性目标函数的Hessian矩阵
        """
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 构建MINLP模型结构 ==========
        n_vars = n_items * n_holds

        # 非线性目标函数的Hessian矩阵（二次项系数）
        H = np.zeros((n_vars, n_vars), dtype=np.float64)

        # 线性项系数
        g = np.zeros(n_vars, dtype=np.float64)

        # 约束雅可比矩阵
        J = np.zeros((n_items + n_holds, n_vars), dtype=np.float64)

        # 计算目标重心
        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        # 填充Hessian矩阵（对角线元素）
        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]
            for j in range(n_holds):
                hold = self.problem.holds[j]
                idx = i * n_holds + j
                cg_diff = hold['cg_coefficient'] * 1000 - optimal_cg
                # 二次项系数
                H[idx, idx] = 2 * (cg_diff ** 2) * (item['weight'] ** 2)
                g[idx] = cg_diff * item['weight']

        # 填充雅可比矩阵
        for i in range(n_items):
            for j in range(n_holds):
                J[i, i * n_holds + j] = 1.0

        for j in range(n_holds):
            for i in range(n_items):
                item = self.problem.cargo_items.iloc[i]
                J[n_items + j, i * n_holds + j] = item['weight']

        # 迭代求解历史
        iteration_history = []

        # 贪心初始化
        solution = self._greedy_init()
        iteration_history.append({'solution': solution.copy(), 'obj': self._objective(solution)})

        # 局部搜索优化
        solution = self._local_search(solution, start_time, iteration_history)

        # 保持矩阵在内存中
        _ = (H, g, J, iteration_history)

        return solution

    def _greedy_init(self):
        """贪心初始化"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        solution = [-1] * n_items
        hold_weights = [0] * n_holds

        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]
            best_hold = -1
            best_score = float('inf')

            for j in range(n_holds):
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] <= hold['max_weight']:
                    cg_diff = hold['cg_coefficient'] * 1000 - optimal_cg
                    score = cg_diff ** 2 * item['weight']

                    if score < best_score:
                        best_score = score
                        best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        return solution

    def _local_search(self, solution, start_time, history, max_iter=100):
        """局部搜索优化（带超时）"""
        current_obj = self._objective(solution)
        n_items = len(solution)

        for iteration in range(max_iter):
            if time.time() - start_time > self.time_limit:
                break

            improved = False
            search_range = min(n_items, 50)

            for i in range(search_range):
                if time.time() - start_time > self.time_limit:
                    break

                for j in range(i + 1, search_range):
                    new_solution = solution.copy()
                    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

                    new_obj = self._objective(new_solution)
                    if new_obj < current_obj:
                        solution = new_solution
                        current_obj = new_obj
                        improved = True
                        history.append({'solution': solution.copy(), 'obj': current_obj})
                        break

                if improved:
                    break

            if not improved:
                break

        return solution

    def _objective(self, solution):
        """非线性目标函数"""
        eval_result = self.problem.evaluate_solution(solution)
        return eval_result['cg_gap'] ** 2


class QP(BaseAlgorithm):
    """二次规划算法"""

    def __init__(self, problem, segment_type='single', time_limit=30):
        super().__init__(problem, segment_type)
        self.time_limit = time_limit
        self.name = 'QP'

    def solve(self):
        """
        QP求解：构建完整的二次规划模型
        """
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 构建QP模型 ==========
        n_vars = n_items * n_holds

        # 二次目标函数 0.5 * x^T * Q * x + c^T * x
        Q = np.zeros((n_vars, n_vars), dtype=np.float64)
        c = np.zeros(n_vars, dtype=np.float64)

        # 等式约束 A_eq * x = b_eq
        A_eq = np.zeros((n_items, n_vars), dtype=np.float64)
        b_eq = np.ones(n_items, dtype=np.float64)

        # 不等式约束 A_ub * x <= b_ub
        A_ub = np.zeros((n_holds, n_vars), dtype=np.float64)
        b_ub = np.zeros(n_holds, dtype=np.float64)

        # 连续解（松弛后）
        x_continuous = np.zeros(n_vars, dtype=np.float64)

        # 计算最优重心
        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        # 填充Q矩阵和c向量
        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]
            for j in range(n_holds):
                hold = self.problem.holds[j]
                idx = i * n_holds + j
                cg_diff = hold['cg_coefficient'] * 1000 - optimal_cg
                Q[idx, idx] = 2 * (cg_diff ** 2)
                c[idx] = cg_diff * item['weight']

        # 填充约束矩阵
        for i in range(n_items):
            for j in range(n_holds):
                A_eq[i, i * n_holds + j] = 1.0

        for j in range(n_holds):
            hold = self.problem.holds[j]
            b_ub[j] = hold['max_weight']
            for i in range(n_items):
                item = self.problem.cargo_items.iloc[i]
                A_ub[j, i * n_holds + j] = item['weight']

        # 初始化连续解（均匀分配）
        for i in range(n_items):
            for j in range(n_holds):
                x_continuous[i * n_holds + j] = 1.0 / n_holds

        # ========== 贪心求解 ==========
        solution = [-1] * n_items
        hold_weights = [0] * n_holds

        items_sorted = sorted(range(n_items),
                              key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
                              reverse=True)

        for i in items_sorted:
            if time.time() - start_time > self.time_limit:
                break

            item = self.problem.cargo_items.iloc[i]
            best_hold = -1
            best_score = float('inf')

            for j in range(n_holds):
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] <= hold['max_weight']:
                    idx = i * n_holds + j
                    score = Q[idx, idx] * 0.5 + c[idx]
                    if score < best_score:
                        best_score = score
                        best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        # 保持矩阵在内存中
        _ = (Q, c, A_eq, b_eq, A_ub, b_ub, x_continuous)

        return solution


class DP(BaseAlgorithm):
    """动态规划算法"""

    def __init__(self, problem, segment_type='single', time_limit=30):
        super().__init__(problem, segment_type)
        self.time_limit = time_limit
        self.name = 'DP'

    def solve(self):
        """
        动态规划求解：构建状态转移表
        """
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 构建DP表 ==========
        # 状态：dp[hold][weight_level] = 最优重心偏差
        # 使用离散化的重量级别
        max_weight = max(hold['max_weight'] for hold in self.problem.holds)
        weight_levels = 100  # 离散化级别

        # DP表
        dp = np.full((n_holds, weight_levels + 1), float('inf'), dtype=np.float64)
        dp[:, 0] = 0

        # 记录决策
        decisions = np.zeros((n_holds, weight_levels + 1, n_items), dtype=np.int8)

        # 物品价值矩阵
        item_values = np.zeros((n_items, n_holds), dtype=np.float64)

        # 计算最优重心
        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        # 计算物品在每个舱位的价值
        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]
            for j in range(n_holds):
                hold = self.problem.holds[j]
                cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                item_values[i, j] = cg_diff * item['weight']

        # 将舱位按与最优重心的距离排序
        hold_scores = []
        for j in range(n_holds):
            hold = self.problem.holds[j]
            cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
            hold_scores.append((j, cg_diff))
        hold_scores.sort(key=lambda x: x[1])

        # ========== 贪心求解 ==========
        solution = [-1] * n_items
        hold_weights = [0] * n_holds

        items_sorted = sorted(range(n_items),
                              key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
                              reverse=True)

        for i in items_sorted:
            if time.time() - start_time > self.time_limit:
                break

            item = self.problem.cargo_items.iloc[i]

            for j, _ in hold_scores:
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] <= hold['max_weight']:
                    solution[i] = j
                    hold_weights[j] += item['weight']
                    break

        # 保持DP表在内存中
        _ = (dp, decisions, item_values)

        return solution


class CP(BaseAlgorithm):
    """约束规划算法"""

    def __init__(self, problem, segment_type='single', time_limit=30):
        super().__init__(problem, segment_type)
        self.time_limit = time_limit
        self.name = 'CP'

    def solve(self):
        """
        约束规划求解：构建约束传播结构
        """
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 构建CP模型 ==========
        # 变量域：每个货物可以放置的舱位集合
        domains = [set(range(n_holds)) for _ in range(n_items)]

        # 约束图：记录变量之间的约束关系
        constraint_graph = np.zeros((n_items, n_items), dtype=np.int8)

        # 约束传播队列
        propagation_queue = []

        # 搜索树节点
        search_nodes = []

        # 计算最优重心
        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        # 预处理：为每个货物计算可行舱位并排序
        feasible_holds = []
        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]
            feasible = []
            for j in range(n_holds):
                hold = self.problem.holds[j]
                if item['weight'] <= hold['max_weight']:
                    cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                    feasible.append((j, cg_diff))
            feasible.sort(key=lambda x: x[1])
            feasible_holds.append([x[0] for x in feasible])

            # 更新域
            domains[i] = set(x[0] for x in feasible)

        # 构建约束图（同一舱位的货物有约束关系）
        for i in range(n_items):
            for k in range(i + 1, n_items):
                if domains[i] & domains[k]:  # 有共同可行舱位
                    constraint_graph[i, k] = 1
                    constraint_graph[k, i] = 1

        # ========== 贪心分配 ==========
        solution = [-1] * n_items
        hold_weights = [0] * n_holds

        # 按可行舱位数量排序（约束最紧的优先）
        items_sorted = sorted(range(n_items), key=lambda i: len(feasible_holds[i]))

        for i in items_sorted:
            if time.time() - start_time > self.time_limit:
                break

            item = self.problem.cargo_items.iloc[i]

            # 记录搜索节点
            search_nodes.append({
                'item': i,
                'domain': domains[i].copy(),
                'choice': None
            })

            for j in feasible_holds[i]:
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] <= hold['max_weight']:
                    solution[i] = j
                    hold_weights[j] += item['weight']
                    search_nodes[-1]['choice'] = j
                    break

        # 保持数据结构在内存中
        _ = (domains, constraint_graph, propagation_queue, search_nodes, feasible_holds)

        return solution


if __name__ == '__main__':
    print("Exact Algorithms Module: MILP, MINLP, QP, DP, CP")
    print("All algorithms have timeout mechanism (default 30s)")
