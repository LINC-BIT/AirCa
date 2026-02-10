# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Cargo Loading Algorithms - Base Class and Utilities
# 算法基类和通用工具
# """
#
# import numpy as np
# import time
# import tracemalloc
# import psutil
# import os
# from abc import ABC, abstractmethod
#
#
# class BaseAlgorithm(ABC):
#     """算法基类"""
#
#     def __init__(self, problem, segment_type='single'):
#         """
#         Args:
#             problem: CargoLoadingProblem instance
#             segment_type: 'single' or 'multi'
#         """
#         self.problem = problem
#         self.segment_type = segment_type
#         self.name = self.__class__.__name__
#
#         # 性能指标
#         self.solve_time = 0
#         self.memory_peak = 0
#         self.io_operations = 0
#
#     @abstractmethod
#     def solve(self):
#         """求解方法，子类必须实现"""
#         pass
#
#     def run_with_metrics(self):
#         """运行算法并收集性能指标"""
#         # 开始内存追踪
#         tracemalloc.start()
#         process = psutil.Process(os.getpid())
#         mem_before = process.memory_info().rss
#
#         # 计时开始
#         start_time = time.time()
#
#         # 运行算法
#         solution = self.solve()
#
#         # 计时结束
#         self.solve_time = time.time() - start_time
#
#         # 内存统计
#         current, peak = tracemalloc.get_traced_memory()
#         tracemalloc.stop()
#         mem_after = process.memory_info().rss
#         self.memory_peak = peak / 1024 / 1024  # MB
#         self.memory_used = (mem_after - mem_before) / 1024 / 1024  # MB
#
#         # 评估解
#         if solution is not None:
#             evaluation = self.problem.evaluate_solution(solution)
#             violations = self.problem.check_constraints(solution)
#         else:
#             evaluation = {
#                 'cg_gap_percent': float('inf'),
#                 'feasible': False,
#                 'revenue': 0
#             }
#             violations = []
#
#         return {
#             'solution': solution,
#             'evaluation': evaluation,
#             'violations': violations,
#             'solve_time': self.solve_time,
#             'memory_peak_mb': self.memory_peak,
#             'memory_used_mb': self.memory_used,
#             'algorithm': self.name
#         }
#
#     def get_objective_value(self, solution):
#         """计算目标函数值"""
#         eval_result = self.problem.evaluate_solution(solution)
#
#         if self.segment_type == 'single':
#             # 单航段：最小化重心偏差
#             return eval_result['cg_gap']
#         else:
#             # 多航段：重心偏差 + 收益（需要权衡）
#             cg_penalty = eval_result['cg_gap'] * 100
#             revenue_reward = eval_result['revenue'] / 1000
#             return cg_penalty - revenue_reward
#
#     def generate_random_solution(self):
#         """生成随机可行解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = []
#         for i in range(n_items):
#             hold_idx = np.random.randint(-1, n_holds)
#             solution.append(hold_idx)
#
#         return solution
#
#     def repair_solution(self, solution):
#         """修复不可行解"""
#         violations = self.problem.check_constraints(solution)
#
#         if not violations:
#             return solution
#
#         solution = list(solution)
#
#         for v in violations:
#             if v['type'] == 'weight':
#                 hold_id = v['hold']
#                 for hold_idx, hold in enumerate(self.problem.holds):
#                     if hold['hold_id'] == hold_id:
#                         items_in_hold = [i for i, h in enumerate(solution) if h == hold_idx]
#                         if items_in_hold:
#                             remove_idx = np.random.choice(items_in_hold)
#                             solution[remove_idx] = -1
#                         break
#
#             elif v['type'] == 'exclusive':
#                 for hold_idx, hold in enumerate(self.problem.holds):
#                     if hold['hold_id'] == v['hold2']:
#                         items_in_hold = [i for i, h in enumerate(solution) if h == hold_idx]
#                         for item_idx in items_in_hold:
#                             solution[item_idx] = -1
#                         break
#
#         return solution
#
#
# class ResultCollector:
#     """结果收集器"""
#
#     def __init__(self):
#         self.results = []
#
#     def add_result(self, result, flight_number=None):
#         """添加单次结果"""
#         result['flight_number'] = flight_number
#         self.results.append(result)
#
#     def get_summary(self):
#         """获取汇总统计"""
#         if not self.results:
#             return None
#
#         algorithms = set(r['algorithm'] for r in self.results)
#         summary = {}
#
#         for algo in algorithms:
#             algo_results = [r for r in self.results if r['algorithm'] == algo]
#
#             cg_gaps = [r['evaluation']['cg_gap_percent'] for r in algo_results
#                        if r['evaluation']['cg_gap_percent'] != float('inf')]
#             times = [r['solve_time'] for r in algo_results]
#             memories = [r['memory_peak_mb'] for r in algo_results]
#             feasible_count = sum(1 for r in algo_results if r['evaluation']['feasible'])
#
#             summary[algo] = {
#                 'avg_cg_gap_percent': np.mean(cg_gaps) if cg_gaps else float('inf'),
#                 'std_cg_gap_percent': np.std(cg_gaps) if cg_gaps else 0,
#                 'avg_solve_time': np.mean(times),
#                 'std_solve_time': np.std(times),
#                 'avg_memory_mb': np.mean(memories),
#                 'std_memory_mb': np.std(memories),
#                 'max_memory_mb': np.max(memories),
#                 'feasible_rate': feasible_count / len(algo_results) * 100,
#                 'n_tests': len(algo_results)
#             }
#
#         return summary
#
#     def to_dataframe(self):
#         """转换为DataFrame"""
#         import pandas as pd
#
#         rows = []
#         for r in self.results:
#             rows.append({
#                 'algorithm': r['algorithm'],
#                 'flight_number': r.get('flight_number', ''),
#                 'cg_gap_percent': r['evaluation']['cg_gap_percent'],
#                 'solve_time_s': r['solve_time'],
#                 'memory_peak_mb': r['memory_peak_mb'],
#                 'feasible': r['evaluation']['feasible'],
#                 'revenue': r['evaluation'].get('revenue', 0)
#             })
#
#         return pd.DataFrame(rows)
#
#     def save_to_csv(self, filepath):
#         """保存到CSV"""
#         df = self.to_dataframe()
#         df.to_csv(filepath, index=False)
#         return df
#
#     def to_latex_table(self, caption="Algorithm Performance Comparison", label="tab:results"):
#         """
#         生成LaTeX格式表格
#         格式: Algorithm & Gap (%) & Memory (MB) & Time (s)
#         每列显示: 平均值 ± 方差
#         """
#         summary = self.get_summary()
#         if not summary:
#             return ""
#
#         # 表头
#         latex = []
#         latex.append(r"\begin{table}[htbp]")
#         latex.append(r"\centering")
#         latex.append(f"\\caption{{{caption}}}")
#         latex.append(f"\\label{{{label}}}")
#         latex.append(r"\begin{tabular}{lccc}")
#         latex.append(r"\toprule")
#         latex.append(r"Algorithm & Gap (\%) & Memory (MB) & Time (s) \\")
#         latex.append(r"\midrule")
#
#         # 按算法名称排序
#         # 精确算法在前，启发式算法在后
#         exact_algos = ['MILP', 'MINLP', 'QP', 'DP', 'CP']
#         heuristic_algos = ['GA', 'PSO', 'CS', 'ACO', 'ABC', 'MBO']
#
#         for algo in exact_algos + heuristic_algos:
#             if algo in summary:
#                 stats = summary[algo]
#                 gap_str = f"${stats['avg_cg_gap_percent']:.2f} \\scriptstyle \\pm {stats['std_cg_gap_percent']:.2f}$"
#                 mem_str = f"${stats['avg_memory_mb']:.2f} \\scriptstyle \\pm {stats['std_memory_mb']:.2f}$"
#                 time_str = f"${stats['avg_solve_time']:.4f} \\scriptstyle \\pm {stats['std_solve_time']:.4f}$"
#
#                 latex.append(f"{algo} & {gap_str} & {mem_str} & {time_str} \\\\")
#
#         latex.append(r"\bottomrule")
#         latex.append(r"\end{tabular}")
#         latex.append(r"\end{table}")
#
#         return "\n".join(latex)
#
#     def save_latex_table(self, filepath, caption="Algorithm Performance Comparison", label="tab:results"):
#         """保存LaTeX表格到文件"""
#         latex_content = self.to_latex_table(caption, label)
#         with open(filepath, 'w', encoding='utf-8') as f:
#             f.write(latex_content)
#         return latex_content
#
#     def print_latex_summary(self, title="Results"):
#         """打印LaTeX格式的汇总"""
#         summary = self.get_summary()
#         if not summary:
#             print("No results to summarize.")
#             return
#
#         print(f"\n% LaTeX Table: {title}")
#         print("% Copy the following to your LaTeX document:")
#         print("-" * 60)
#         print(self.to_latex_table(caption=title, label=f"tab:{title.lower().replace(' ', '_')}"))
#         print("-" * 60)


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cargo Loading Algorithms - Base Class and Utilities
算法基类和通用工具

修改说明：
1. 添加舱位数量约束（每个舱位最多1个ULD）
2. 添加ULD类型匹配约束
3. 修改repair_solution()支持新约束
"""

import numpy as np
import time
import tracemalloc
import psutil
import os
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """算法基类"""

    def __init__(self, problem, segment_type='single'):
        """
        Args:
            problem: CargoLoadingProblem instance
            segment_type: 'single' or 'multi'
        """
        self.problem = problem
        self.segment_type = segment_type
        self.name = self.__class__.__name__

        # 性能指标
        self.solve_time = 0
        self.memory_peak = 0
        self.io_operations = 0

    @abstractmethod
    def solve(self):
        """求解方法，子类必须实现"""
        pass

    def run_with_metrics(self):
        """运行算法并收集性能指标"""
        # 开始内存追踪
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        # 计时开始
        start_time = time.time()

        # 运行算法
        solution = self.solve()

        # 计时结束
        self.solve_time = time.time() - start_time

        # 内存统计
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_after = process.memory_info().rss
        self.memory_peak = peak / 1024 / 1024  # MB
        self.memory_used = (mem_after - mem_before) / 1024 / 1024  # MB

        # 评估解
        if solution is not None:
            evaluation = self.problem.evaluate_solution(solution)
            violations = self.problem.check_constraints(solution)
        else:
            evaluation = {
                'cg_gap_percent': float('inf'),
                'feasible': False,
                'revenue': 0
            }
            violations = []

        return {
            'solution': solution,
            'evaluation': evaluation,
            'violations': violations,
            'solve_time': self.solve_time,
            'memory_peak_mb': self.memory_peak,
            'memory_used_mb': self.memory_used,
            'algorithm': self.name
        }

    def get_objective_value(self, solution):
        """计算目标函数值"""
        eval_result = self.problem.evaluate_solution(solution)

        if self.segment_type == 'single':
            # 单航段：最小化重心偏差
            return eval_result['cg_gap']
        else:
            # 多航段：重心偏差 + 收益（需要权衡）
            cg_penalty = eval_result['cg_gap'] * 100
            revenue_reward = eval_result['revenue'] / 1000
            return cg_penalty - revenue_reward

    def is_hold_compatible(self, item, hold):
        """
        检查货物是否可以放入舱位（ULD类型匹配）

        Args:
            item: 货物数据（DataFrame row或dict）
            hold: 舱位数据（dict）

        Returns:
            bool: 是否兼容
        """
        # 获取货物的ULD类型
        if hasattr(item, 'get'):
            item_uld_type = item.get('uld_type', '')
        else:
            item_uld_type = getattr(item, 'uld_type', '') if hasattr(item, 'uld_type') else ''

        # 获取舱位允许的ULD类型
        allowed_types = hold.get('uld_types', [])

        # 如果舱位没有类型限制，或者货物没有类型信息，则认为兼容
        if not allowed_types or not item_uld_type:
            return True

        # 检查货物类型是否在允许列表中
        return item_uld_type in allowed_types

    def generate_random_solution(self):
        """生成随机可行解（考虑每个舱位只能放1个ULD）"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        solution = [-1] * n_items
        hold_occupied = [False] * n_holds  # 记录舱位是否已被占用

        # 随机打乱货物顺序
        item_order = list(range(n_items))
        np.random.shuffle(item_order)

        for i in item_order:
            item = self.problem.cargo_items.iloc[i]

            # 找到可用的舱位（未占用、类型匹配、重量不超限）
            available_holds = []
            for j in range(n_holds):
                if hold_occupied[j]:
                    continue
                hold = self.problem.holds[j]
                if item['weight'] > hold['max_weight']:
                    continue
                if not self.is_hold_compatible(item, hold):
                    continue
                available_holds.append(j)

            if available_holds:
                hold_idx = np.random.choice(available_holds)
                solution[i] = hold_idx
                hold_occupied[hold_idx] = True

        return solution

    def repair_solution(self, solution):
        """
        修复不可行解

        修复顺序：
        1. 舱位数量超限（每个舱位只能放1个ULD）
        2. ULD类型不匹配
        3. 重量超限
        4. 互斥舱位冲突
        """
        solution = list(solution)
        n_holds = self.problem.n_holds

        # ========== 1. 修复舱位数量超限（每个舱位最多1个ULD） ==========
        hold_items = {}  # hold_idx -> list of item_idx
        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0:
                if hold_idx not in hold_items:
                    hold_items[hold_idx] = []
                hold_items[hold_idx].append(item_idx)

        for hold_idx, items in hold_items.items():
            if len(items) > 1:
                # 保留重量最大的，其余移除
                items_sorted = sorted(items,
                                      key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
                                      reverse=True)
                for item_idx in items_sorted[1:]:  # 移除多余的
                    solution[item_idx] = -1

        # ========== 2. 修复ULD类型不匹配 ==========
        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0:
                item = self.problem.cargo_items.iloc[item_idx]
                hold = self.problem.holds[hold_idx]

                if not self.is_hold_compatible(item, hold):
                    solution[item_idx] = -1  # 移除不匹配的

        # ========== 3. 修复重量超限 ==========
        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0:
                item = self.problem.cargo_items.iloc[item_idx]
                hold = self.problem.holds[hold_idx]

                if item['weight'] > hold['max_weight']:
                    solution[item_idx] = -1

        # ========== 4. 修复互斥舱位冲突 ==========
        used_holds = set(h for h in solution if h >= 0)
        holds_to_clear = set()

        for hold_idx in used_holds:
            hold = self.problem.holds[hold_idx]
            exclusive = hold.get('exclusive_holds', [])
            for exc_hold_id in exclusive:
                if exc_hold_id:
                    for other_idx, other_hold in enumerate(self.problem.holds):
                        if other_hold['hold_id'] == exc_hold_id and other_idx in used_holds:
                            # 随机选择一个清除
                            holds_to_clear.add(other_idx)

        for item_idx, hold_idx in enumerate(solution):
            if hold_idx in holds_to_clear:
                solution[item_idx] = -1

        return solution

    def get_available_holds(self, item, hold_occupied):
        """
        获取货物可用的舱位列表

        Args:
            item: 货物数据
            hold_occupied: 舱位占用状态列表

        Returns:
            list: 可用舱位索引列表
        """
        available = []
        for j in range(self.problem.n_holds):
            # 检查是否已被占用
            if hold_occupied[j]:
                continue

            hold = self.problem.holds[j]

            # 检查重量
            if item['weight'] > hold['max_weight']:
                continue

            # 检查ULD类型匹配
            if not self.is_hold_compatible(item, hold):
                continue

            available.append(j)

        return available


class ResultCollector:
    """结果收集器"""

    def __init__(self):
        self.results = []

    def add_result(self, result, flight_number=None):
        """添加单次结果"""
        result['flight_number'] = flight_number
        self.results.append(result)

    def get_summary(self):
        """获取汇总统计"""
        if not self.results:
            return None

        algorithms = set(r['algorithm'] for r in self.results)
        summary = {}

        for algo in algorithms:
            algo_results = [r for r in self.results if r['algorithm'] == algo]

            cg_gaps = [r['evaluation']['cg_gap_percent'] for r in algo_results
                       if r['evaluation']['cg_gap_percent'] != float('inf')]
            times = [r['solve_time'] for r in algo_results]
            memories = [r['memory_peak_mb'] for r in algo_results]
            feasible_count = sum(1 for r in algo_results if r['evaluation']['feasible'])

            summary[algo] = {
                'avg_cg_gap_percent': np.mean(cg_gaps) if cg_gaps else float('inf'),
                'std_cg_gap_percent': np.std(cg_gaps) if cg_gaps else 0,
                'avg_solve_time': np.mean(times),
                'std_solve_time': np.std(times),
                'avg_memory_mb': np.mean(memories),
                'std_memory_mb': np.std(memories),
                'max_memory_mb': np.max(memories),
                'feasible_rate': feasible_count / len(algo_results) * 100,
                'n_tests': len(algo_results)
            }

        return summary

    def to_dataframe(self):
        """转换为DataFrame"""
        import pandas as pd

        rows = []
        for r in self.results:
            rows.append({
                'algorithm': r['algorithm'],
                'flight_number': r.get('flight_number', ''),
                'cg_gap_percent': r['evaluation']['cg_gap_percent'],
                'solve_time_s': r['solve_time'],
                'memory_peak_mb': r['memory_peak_mb'],
                'feasible': r['evaluation']['feasible'],
                'revenue': r['evaluation'].get('revenue', 0)
            })

        return pd.DataFrame(rows)

    def save_to_csv(self, filepath):
        """保存到CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        return df

    def to_latex_table(self, caption="Algorithm Performance Comparison", label="tab:results"):
        """
        生成LaTeX格式表格
        格式: Algorithm & Gap (%) & Memory (MB) & Time (s)
        每列显示: 平均值 ± 方差
        """
        summary = self.get_summary()
        if not summary:
            return ""

        # 表头
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append(r"\begin{tabular}{lccc}")
        latex.append(r"\toprule")
        latex.append(r"Algorithm & Gap (\%) & Memory (MB) & Time (s) \\")
        latex.append(r"\midrule")

        # 按算法名称排序
        # 精确算法在前，启发式算法在后
        exact_algos = ['MILP', 'MINLP', 'QP', 'DP', 'CP']
        heuristic_algos = ['GA', 'PSO', 'CS', 'ACO', 'ABC', 'MBO']

        for algo in exact_algos + heuristic_algos:
            if algo in summary:
                stats = summary[algo]
                gap_str = f"${stats['avg_cg_gap_percent']:.2f} \\scriptstyle \\pm {stats['std_cg_gap_percent']:.2f}$"
                mem_str = f"${stats['avg_memory_mb']:.2f} \\scriptstyle \\pm {stats['std_memory_mb']:.2f}$"
                time_str = f"${stats['avg_solve_time']:.4f} \\scriptstyle \\pm {stats['std_solve_time']:.4f}$"

                latex.append(f"{algo} & {gap_str} & {mem_str} & {time_str} \\\\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

    def save_latex_table(self, filepath, caption="Algorithm Performance Comparison", label="tab:results"):
        """保存LaTeX表格到文件"""
        latex_content = self.to_latex_table(caption, label)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        return latex_content

    def print_latex_summary(self, title="Results"):
        """打印LaTeX格式的汇总"""
        summary = self.get_summary()
        if not summary:
            print("No results to summarize.")
            return

        print(f"\n% LaTeX Table: {title}")
        print("% Copy the following to your LaTeX document:")
        print("-" * 60)
        print(self.to_latex_table(caption=title, label=f"tab:{title.lower().replace(' ', '_')}"))
        print("-" * 60)
