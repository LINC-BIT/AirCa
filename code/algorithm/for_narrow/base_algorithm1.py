#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cargo Loading Algorithms - Base Class and Utilities
算法基类和通用工具
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

    def generate_random_solution(self):
        """生成随机可行解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        solution = []
        for i in range(n_items):
            hold_idx = np.random.randint(-1, n_holds)
            solution.append(hold_idx)

        return solution

    def repair_solution(self, solution):
        """修复不可行解"""
        violations = self.problem.check_constraints(solution)

        if not violations:
            return solution

        solution = list(solution)

        for v in violations:
            if v['type'] == 'weight':
                hold_id = v['hold']
                for hold_idx, hold in enumerate(self.problem.holds):
                    if hold['hold_id'] == hold_id:
                        items_in_hold = [i for i, h in enumerate(solution) if h == hold_idx]
                        if items_in_hold:
                            remove_idx = np.random.choice(items_in_hold)
                            solution[remove_idx] = -1
                        break

            elif v['type'] == 'exclusive':
                for hold_idx, hold in enumerate(self.problem.holds):
                    if hold['hold_id'] == v['hold2']:
                        items_in_hold = [i for i, h in enumerate(solution) if h == hold_idx]
                        for item_idx in items_in_hold:
                            solution[item_idx] = -1
                        break

        return solution


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
