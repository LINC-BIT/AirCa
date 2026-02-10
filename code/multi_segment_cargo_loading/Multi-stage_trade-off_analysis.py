import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import interpolate
import glob
import time
import traceback


def _ensure_algorithm_on_path():
    """Add the project code root to sys.path so 'algorithm' package is importable."""
    # 1) If already importable, nothing to do
    try:
        import algorithm  # noqa: F401
        return
    except ImportError:
        pass

    # 2) Walk up from this file's directory
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):  # at most 5 levels up
        if os.path.isdir(os.path.join(d, 'algorithm')):
            if d not in sys.path:
                sys.path.insert(0, d)
                print(f"[path] Added '{d}' to sys.path for algorithm imports")
            return
        d = os.path.dirname(d)

    # 3) Fallback: warn but don't crash yet (will crash on actual import)
    print("[WARNING] Could not auto-detect 'algorithm' package location. "
          "Use --code-path to specify the directory containing 'algorithm/'.")

_ensure_algorithm_on_path()

# ==================== 路径配置 ====================
BENCHMARK_PATH = r'G:\AirCa\code\aircraft_data'
CARGO_DATA_PATH = r'G:\loading_benchmark\bakFlightLoadData\bakFlightLoadData'
OUTPUT_PATH = r'G:\loading_benchmark\AirCa_output\multi_stage4'

# 机型配置
AIRCRAFT_CONFIGS = {
'B777': {'type': 'widebody', 'folder': 'B777', 'fleet_pattern': ['B777', '777', '77A', '77B', '77W']},
'A320': {'type': 'narrowbody', 'folder': 'A320', 'fleet_pattern': ['A320', '320', '32A', '32B', '32N']},

}

# 多阶段权重比例
WEIGHT_RATIOS = [(1.0, 0.0), (0.0, 1.0) ]


class MultiAircraftDataLoader:
    """多机型数据加载器"""

    def __init__(self, aircraft_type):
        if aircraft_type not in AIRCRAFT_CONFIGS:
            raise ValueError(f"不支持的机型: {aircraft_type}")

        self.aircraft_type = aircraft_type
        self.config = AIRCRAFT_CONFIGS[aircraft_type]
        self.base_path = os.path.join(BENCHMARK_PATH, self.config['folder'])
        self.cargo_data_path = CARGO_DATA_PATH

        self.cargo_holds = None
        self.flight_params = None
        self.cg_limits = None
        self.cargo_data = None

    def load_all(self):
        """加载所有数据"""
        self.load_cargo_holds()
        self.load_flight_params()
        self.load_cg_limits()
        self.load_cargo_data()
        return self

    def load_cargo_holds(self):
        """加载舱位信息"""
        folder = self.config['folder']
        filepath = os.path.join(self.base_path, f'{folder}.csv')

        if not os.path.exists(filepath):
            for f in os.listdir(self.base_path):
                if f.endswith('.csv') and 'zfw' not in f.lower():
                    filepath = os.path.join(self.base_path, f)
                    break

        df = pd.read_csv(filepath, header=None)

        cargo_holds = []
        for _, row in df.iterrows():
            hold_id = str(row[1]).strip() if pd.notna(row[1]) else None
            if hold_id is None or hold_id == 'nan':
                continue

            exclusive = str(row[2]).strip() if pd.notna(row[2]) else ''
            uld_types = str(row[3]).strip() if pd.notna(row[3]) else ''
            max_weight = float(row[4]) if pd.notna(row[4]) else 0

            cg_coef = None
            for i in range(len(row) - 1, -1, -1):
                if pd.notna(row[i]) and row[i] != '':
                    try:
                        cg_coef = float(row[i])
                        break
                    except:
                        continue

            arm = None
            for i in [8, 9, 10]:
                if i < len(row) and pd.notna(row[i]):
                    try:
                        arm = float(row[i])
                        break
                    except:
                        continue

            # 解析ULD类型列表和互斥舱位
            uld_type_list = [x.strip() for x in uld_types.split('/') if x.strip() and x.strip() != '////']
            exclusive_list = [x.strip() for x in exclusive.split('/') if x.strip() and x.strip() != '////']

            cargo_holds.append({
                'hold_id': hold_id,
                'exclusive_holds': exclusive_list,
                'uld_types': uld_type_list,
                'max_weight': max_weight,
                'arm': arm,
                'cg_coefficient': cg_coef
            })

        self.cargo_holds = pd.DataFrame(cargo_holds)
        return self.cargo_holds

    def load_flight_params(self):
        """加载航班参数"""
        filepath = os.path.join(self.base_path, '航班参数.csv')
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='gbk')
            except:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except:
                    self.flight_params = {'initial_weight': 50000, 'initial_cg': 0}
                    return self.flight_params
        else:
            filepath = os.path.join(self.base_path, '航班参数.xlsx')
            if os.path.exists(filepath):
                df = pd.read_excel(filepath)
            else:
                self.flight_params = {'initial_weight': 50000, 'initial_cg': 0}
                return self.flight_params

        row = df.iloc[0]

        try:
            weight1 = float(row.get('dryOperatingWeight', row.get('干操作重量', 50000)))
            cg1 = float(row.get('dryOperatingCenter', row.get('干操作重心', 0)))
            weight2 = float(row.get('passengerWeight', row.get('旅客重量', 10000)))
            cg2 = float(row.get('passengerCenter', row.get('旅客重心', 0)))
        except:
            weight1, cg1, weight2, cg2 = 50000, 0, 10000, 0

        self.flight_params = {
            'initial_weight': weight1 + weight2,
            'initial_cg': cg1 + cg2
        }

        return self.flight_params

    def load_cg_limits(self):
        """加载CG限制"""
        aft_file = os.path.join(self.base_path, 'stdZfw_a.csv')
        fwd_file = os.path.join(self.base_path, 'stdZfw_f.csv')

        if os.path.exists(aft_file) and os.path.exists(fwd_file):
            df_aft = pd.read_csv(aft_file, header=None)
            df_fwd = pd.read_csv(fwd_file, header=None)

            weights_aft = df_aft.iloc[:, 1].values.astype(float)
            cg_aft = df_aft.iloc[:, 3].values.astype(float)
            weights_fwd = df_fwd.iloc[:, 1].values.astype(float)
            cg_fwd = df_fwd.iloc[:, 3].values.astype(float)

            aft_interp = interpolate.interp1d(weights_aft, cg_aft, kind='linear', fill_value='extrapolate')
            fwd_interp = interpolate.interp1d(weights_fwd, cg_fwd, kind='linear', fill_value='extrapolate')
        else:
            aft_interp = lambda w: -0.1
            fwd_interp = lambda w: 0.1

        self.cg_limits = {
            'aft_limit': aft_interp,
            'fwd_limit': fwd_interp
        }

        return self.cg_limits

    def load_cargo_data(self):
        """加载货物数据"""
        cargo_files = glob.glob(os.path.join(self.cargo_data_path, 'BAKFLGITH_LOADDATA*.csv'))

        if not cargo_files:
            raise FileNotFoundError(f"未找到货物数据: {self.cargo_data_path}")

        all_data = []
        for cargo_file in cargo_files:
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(cargo_file, encoding=encoding, header=None)
                    all_data.append(df)
                    break
                except:
                    continue

        df = pd.concat(all_data, ignore_index=True)

        df['flight_number'] = df.iloc[:, 0].astype(str)
        df['fleet'] = df.iloc[:, 1].astype(str).str.upper()
        df['destination'] = df.iloc[:, 2].astype(str).str.upper()
        df['weight'] = pd.to_numeric(df.iloc[:, 3], errors='coerce').fillna(0)
        df['content_type'] = df.iloc[:, 4].astype(str).str.upper()

        # 第8列是ULD类型
        if df.shape[1] > 7:
            df['uld_type'] = df.iloc[:, 7].astype(str).str.strip()
            df['uld_type'] = df['uld_type'].apply(lambda x: '' if x == 'nan' or pd.isna(x) else x)
        else:
            df['uld_type'] = ''

        if df.shape[1] > 11:
            df['volume'] = pd.to_numeric(df.iloc[:, 11], errors='coerce').fillna(0)
        else:
            df['volume'] = 0

        # 判断是否为散货
        df['is_bulk'] = df['uld_type'].apply(lambda x: x == '' or x == 'nan' or pd.isna(x))

        # 筛选机型
        patterns = self.config['fleet_pattern']
        mask = df['fleet'].apply(lambda x: any(p in str(x).upper() for p in patterns))
        df_filtered = df[mask].copy()

        # 判断单/多航段
        flight_dest_count = df_filtered.groupby('flight_number')['destination'].nunique()
        multi_flights = set(flight_dest_count[flight_dest_count > 1].index)
        df_filtered['is_multi_segment'] = df_filtered['flight_number'].isin(multi_flights)

        # 计算每个航班的货物总重
        flight_weights = df_filtered.groupby('flight_number')['weight'].sum().reset_index()
        flight_weights.columns = ['flight_number', 'total_weight']
        df_filtered = df_filtered.merge(flight_weights, on='flight_number')

        self.cargo_data = df_filtered
        print(f"    加载货物数据: {len(df_filtered)} 条 ({self.aircraft_type})")
        return self.cargo_data

    def get_single_flights(self, n_flights=100):
        """获取单航段航班（按货物量排序）"""
        df = self.cargo_data[~self.cargo_data['is_multi_segment']]
        flights = df.groupby('flight_number')['weight'].sum().reset_index()
        flights = flights.sort_values('weight', ascending=False)
        return flights['flight_number'].head(n_flights).tolist()

    def get_multi_flights(self, n_flights=100):
        """获取多航段航班（按货物量排序）"""
        df = self.cargo_data[self.cargo_data['is_multi_segment']]
        flights = df.groupby('flight_number')['weight'].sum().reset_index()
        flights = flights.sort_values('weight', ascending=False)
        return flights['flight_number'].head(n_flights).tolist()

    def get_paired_flights(self, n_pairs=50):
        """获取配对的单/多航段航班（相似货物量）"""
        single_df = self.cargo_data[~self.cargo_data['is_multi_segment']]
        multi_df = self.cargo_data[self.cargo_data['is_multi_segment']]

        single_weights = single_df.groupby('flight_number')['weight'].sum().reset_index()
        multi_weights = multi_df.groupby('flight_number')['weight'].sum().reset_index()

        if len(single_weights) == 0 or len(multi_weights) == 0:
            return []

        pairs = []
        used_single = set()

        for _, multi_row in multi_weights.iterrows():
            multi_flight = multi_row['flight_number']
            multi_weight = multi_row['weight']

            available = single_weights[~single_weights['flight_number'].isin(used_single)]
            if len(available) == 0:
                break

            available = available.copy()
            available['diff'] = abs(available['weight'] - multi_weight)
            best_match = available.loc[available['diff'].idxmin()]

            single_flight = best_match['flight_number']
            used_single.add(single_flight)

            pairs.append({
                'single_flight': single_flight,
                'single_weight': best_match['weight'],
                'multi_flight': multi_flight,
                'multi_weight': multi_weight,
                'weight_diff': best_match['diff']
            })

            if len(pairs) >= n_pairs:
                break

        return pairs

    def get_flight_cargo(self, flight_number):
        """获取指定航班的货物"""
        return self.cargo_data[self.cargo_data['flight_number'] == flight_number].copy()


class CargoLoadingProblemMultiStage:
    """
    支持多阶段优化的货物装载问题

    完整约束实现（特别针对宽体机）：
    1. 舱位容量约束：每个舱位最多装载一个ULD
    2. ULD类型匹配约束：货物ULD类型必须与舱位允许类型匹配
    3. 互斥舱位约束：互斥舱位不能同时使用
    4. 重量约束：舱位载重不超过限制
    """

    def __init__(self, cargo_holds, flight_params, cg_limits, cargo_items,
                 segment_type='single', cg_weight=0.5, revenue_weight=0.5,
                 is_widebody=True):
        self.cargo_holds = cargo_holds
        self.flight_params = flight_params
        self.cg_limits = cg_limits
        self.cargo_items = cargo_items
        self.segment_type = segment_type
        self.cg_weight = cg_weight
        self.revenue_weight = revenue_weight
        self.is_widebody = is_widebody

        self.holds = self._get_unique_holds()
        self.n_holds = len(self.holds)
        self.n_items = len(cargo_items)

        self.initial_weight = flight_params['initial_weight']
        self.initial_cg = flight_params['initial_cg']

        # 预计算用于归一化的参数
        self._max_possible_revenue = self._estimate_max_revenue()

    def _estimate_max_revenue(self):
        """估算最大可能的revenue，用于归一化"""
        total = 0
        for idx in range(len(self.cargo_items)):
            weight = self.cargo_items.iloc[idx]['weight']
            # 使用简化的费率估算
            rate = 5.44 if weight >= 1000 else (6.27 if weight >= 500 else 9.07)
            total += max(70, weight * rate)
        return max(total, 1)  # 避免除零

    def _get_unique_holds(self):
        holds = []
        seen = set()
        for _, row in self.cargo_holds.iterrows():
            hold_id = row['hold_id']
            if hold_id not in seen:
                holds.append(row.to_dict())
                seen.add(hold_id)
        return holds

    def is_hold_compatible(self, item, hold):
        """
        检查货物是否与舱位的ULD类型兼容

        Args:
            item: 货物数据
            hold: 舱位数据

        Returns:
            bool: 是否兼容
        """
        # 获取货物的ULD类型
        if hasattr(item, 'uld_type'):
            item_uld_type = item.uld_type
        elif isinstance(item, dict):
            item_uld_type = item.get('uld_type', '')
        else:
            item_uld_type = ''

        item_uld_type = str(item_uld_type).strip() if item_uld_type else ''

        # 获取舱位允许的ULD类型
        allowed_types = hold.get('uld_types', [])

        # 如果舱位没有限制ULD类型，或者货物没有ULD类型，认为兼容
        if not allowed_types or not item_uld_type:
            return True

        # 检查货物的ULD类型是否在允许列表中
        return item_uld_type in allowed_types

    def get_optimal_cg(self, total_weight):
        aft = float(self.cg_limits['aft_limit'](total_weight))
        fwd = float(self.cg_limits['fwd_limit'](total_weight))
        optimal = aft + (fwd - aft) * (1 / 3)
        return optimal, aft, fwd

    def calculate_cg(self, solution):
        total_moment = self.initial_cg
        total_weight = self.initial_weight

        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0 and hold_idx < len(self.holds):
                item = self.cargo_items.iloc[item_idx]
                hold = self.holds[hold_idx]
                weight = item['weight']
                cg_coef = hold['cg_coefficient']

                if cg_coef is not None:
                    total_weight += weight
                    total_moment += weight * cg_coef

        return total_moment, total_weight

    def calculate_cargo_revenue(self, weight):
        """
        根据货物重量计算运费 (基于TPE-DFW航线费率表)

        费率表 (Rate-class intervals):
        - 最低收费(M): 70 US$
        - ~44 kg: 12.66 US$/kg
        - 45-99 kg: 9.74 US$/kg
        - 100-299 kg: 9.07 US$/kg
        - 300-499 kg: 7.16 US$/kg
        - 500-999 kg: 6.27 US$/kg
        - 1000+ kg: 5.44 US$/kg

        Args:
            weight: 货物重量 (kg)

        Returns:
            revenue: 运费 (US$)
        """
        # 最低收费
        MIN_CHARGE = 70.0

        # 费率表: (上限重量, 费率)
        RATE_TABLE = [
            (44, 12.66),
            (99, 9.74),
            (299, 9.07),
            (499, 7.16),
            (999, 6.27),
            (float('inf'), 5.44)
        ]

        # 根据重量确定费率
        rate = RATE_TABLE[-1][1]  # 默认最高档费率
        for upper_limit, r in RATE_TABLE:
            if weight <= upper_limit:
                rate = r
                break

        # 计算运费，取最低收费和按重量计算的较大值
        revenue = max(MIN_CHARGE, weight * rate)

        return revenue

    def calculate_profit(self, gross_revenue, cg_gap_percent):
        """
        根据CG偏差计算实际利润

        CG偏离越大，燃油效率越低，利润越低

        Profit = Gross Revenue × (1 - penalty_factor)

        penalty_factor 随 cg_gap_percent 增大:
        - cg_gap = 0%: penalty = 0% (满利润)
        - cg_gap = 10%: penalty ≈ 5%
        - cg_gap = 50%: penalty ≈ 25%
        - cg_gap = 100%: penalty ≈ 50%

        使用公式: penalty = 0.5 × (1 - exp(-cg_gap_percent / 50))
        这样penalty最大不超过50%，且随gap增大逐渐饱和

        Args:
            gross_revenue: 总运费 (US$)
            cg_gap_percent: CG偏差百分比

        Returns:
            profit: 实际利润 (US$)
        """
        import math

        # 惩罚因子：CG gap越大，惩罚越大，最大50%
        penalty_factor = 0.5 * (1 - math.exp(-cg_gap_percent / 50))

        # 利润 = 总运费 × (1 - 惩罚因子)
        profit = gross_revenue * (1 - penalty_factor)

        return profit

    def evaluate_solution(self, solution):
        cg, total_weight = self.calculate_cg(solution)
        optimal_cg, aft_limit, fwd_limit = self.get_optimal_cg(total_weight)

        cg_gap = abs(cg - optimal_cg)
        cg_gap_percent = cg_gap / abs(optimal_cg) * 100 if optimal_cg != 0 else 0

        cg_feasible = aft_limit <= cg <= fwd_limit

        # 检查所有约束
        violations = self.check_constraints(solution)
        feasible = cg_feasible and len(violations) == 0

        # 计算总运费 (基于费率表)
        gross_revenue = 0
        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0:
                cargo_weight = self.cargo_items.iloc[item_idx]['weight']
                gross_revenue += self.calculate_cargo_revenue(cargo_weight)

        # 计算实际利润 (考虑CG偏差的惩罚)
        profit = self.calculate_profit(gross_revenue, cg_gap_percent)

        return {
            'cg': cg,
            'total_weight': total_weight,
            'optimal_cg': optimal_cg,
            'cg_gap': cg_gap,
            'cg_gap_percent': cg_gap_percent,
            'aft_limit': aft_limit,
            'fwd_limit': fwd_limit,
            'feasible': feasible,
            'gross_revenue': gross_revenue,
            'revenue': profit  # profit作为最终revenue返回
        }

    def get_weighted_score(self, solution):
        """
        计算加权目标函数分数，用于多目标优化

        Score = cg_weight × CG_score + revenue_weight × Revenue_score

        其中:
        - CG_score = 1 - (cg_gap_percent / 100)，越小越好，归一化到[0,1]
        - Revenue_score = profit / max_possible_revenue，越大越好，归一化到[0,1]

        最终Score越高越好

        Args:
            solution: 装载方案

        Returns:
            score: 加权分数 (越高越好)
        """
        eval_result = self.evaluate_solution(solution)

        # CG分数：gap越小分数越高
        cg_gap_percent = eval_result['cg_gap_percent']
        cg_score = max(0, 1 - cg_gap_percent / 100)  # 归一化到[0,1]

        # Revenue分数：profit越高分数越高
        profit = eval_result['revenue']
        revenue_score = profit / self._max_possible_revenue if self._max_possible_revenue > 0 else 0
        revenue_score = min(1, revenue_score)  # 确保不超过1

        # 加权综合分数
        weighted_score = self.cg_weight * cg_score + self.revenue_weight * revenue_score

        return weighted_score

    def get_objective_value(self, solution):
        """
        计算目标函数值，用于算法优化（值越小越好）

        Objective = cg_weight × normalized_cg_gap - revenue_weight × normalized_revenue

        注意：返回值越小越好（用于最小化问题）

        Args:
            solution: 装载方案

        Returns:
            objective: 目标函数值 (越小越好)
        """
        eval_result = self.evaluate_solution(solution)

        # CG项：gap越大值越大（要最小化）
        cg_gap_percent = eval_result['cg_gap_percent']
        normalized_cg = cg_gap_percent / 100  # 归一化

        # Revenue项：profit越高值越大（要最大化，所以取负）
        profit = eval_result['revenue']
        normalized_revenue = profit / self._max_possible_revenue if self._max_possible_revenue > 0 else 0

        # 综合目标：cg_weight × cg - revenue_weight × revenue
        # 值越小越好
        objective = self.cg_weight * normalized_cg - self.revenue_weight * normalized_revenue

        return objective

    def check_constraints(self, solution):
        """
        检查所有约束

        约束类型：
        1. capacity - 舱位容量约束（每个舱位最多1个ULD）- 仅宽体机
        2. uld_type - ULD类型匹配约束 - 仅宽体机
        3. weight - 舱位重量约束
        4. exclusive - 互斥舱位约束
        """
        violations = []
        hold_weights = {i: 0 for i in range(self.n_holds)}
        hold_items = {i: [] for i in range(self.n_holds)}

        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0 and hold_idx < self.n_holds:
                item = self.cargo_items.iloc[item_idx]
                hold_weights[hold_idx] += item['weight']
                hold_items[hold_idx].append(item_idx)

        # ========== 1. 舱位容量约束（每个舱位最多1个ULD）- 仅宽体机 ==========
        if self.is_widebody:
            for hold_idx, item_list in hold_items.items():
                if len(item_list) > 1:
                    violations.append({
                        'type': 'capacity',
                        'hold': self.holds[hold_idx]['hold_id'],
                        'hold_idx': hold_idx,
                        'actual': len(item_list),
                        'limit': 1,
                        'items': item_list
                    })

        # ========== 2. ULD类型匹配约束 - 仅宽体机 ==========
        if self.is_widebody:
            for item_idx, hold_idx in enumerate(solution):
                if hold_idx >= 0 and hold_idx < self.n_holds:
                    item = self.cargo_items.iloc[item_idx]
                    hold = self.holds[hold_idx]

                    if not self.is_hold_compatible(item, hold):
                        item_uld_type = item.get('uld_type', '') if isinstance(item, dict) else getattr(item,
                                                                                                        'uld_type', '')
                        allowed_types = hold.get('uld_types', [])

                        violations.append({
                            'type': 'uld_type',
                            'item': item_idx,
                            'item_type': str(item_uld_type),
                            'hold': hold['hold_id'],
                            'hold_idx': hold_idx,
                            'allowed_types': allowed_types
                        })

        # ========== 3. 重量约束 ==========
        for hold_idx, weight in hold_weights.items():
            if weight > self.holds[hold_idx]['max_weight']:
                violations.append({
                    'type': 'weight',
                    'hold': self.holds[hold_idx]['hold_id'],
                    'hold_idx': hold_idx,
                    'actual': weight,
                    'limit': self.holds[hold_idx]['max_weight']
                })

        # ========== 4. 互斥舱位约束 ==========
        checked_pairs = set()
        for hold_idx, hold in enumerate(self.holds):
            if hold_items[hold_idx] and hold.get('exclusive_holds'):
                for excl_hold_id in hold['exclusive_holds']:
                    if excl_hold_id:
                        for other_idx, other_hold in enumerate(self.holds):
                            if other_hold['hold_id'] == excl_hold_id and hold_items[other_idx]:
                                pair = tuple(sorted([hold_idx, other_idx]))
                                if pair not in checked_pairs:
                                    violations.append({
                                        'type': 'exclusive',
                                        'hold1': hold['hold_id'],
                                        'hold1_idx': hold_idx,
                                        'hold2': excl_hold_id,
                                        'hold2_idx': other_idx
                                    })
                                    checked_pairs.add(pair)

        return violations


class MultiStageExperimentRunner:
    """多阶段优化实验运行器"""

    def __init__(self):
        self.output_path = OUTPUT_PATH
        os.makedirs(self.output_path, exist_ok=True)
        self.start_time = None

    def get_algorithms_narrowbody(self, time_limit=30):
        """获取窄体机算法"""
        from algorithm.for_narrow.exact_algorithms1 import MILP, MINLP, QP, DP, CP
        from algorithm.for_narrow.heuristic_algorithms1 import GA, PSO, CS, ACO, ABC, MBO
        return [MILP, MINLP, QP, DP, CP, GA, PSO, CS, ACO, ABC, MBO]

    def get_algorithms_widebody(self, time_limit=30):
        """获取宽体机算法"""
        from algorithm.for_wide.exact_algorithms import MILP, MINLP, QP, DP, CP
        from algorithm.for_wide.heuristic_algorithms import GA, PSO, CS, ACO, ABC, MBO
        return [MILP, MINLP, QP, DP, CP, GA, PSO, CS, ACO, ABC, MBO]

    def get_algorithms(self, aircraft_type, time_limit=30):
        """根据机型获取对应算法"""
        if AIRCRAFT_CONFIGS[aircraft_type]['type'] == 'narrowbody':
            return self.get_algorithms_narrowbody(time_limit)
        else:
            return self.get_algorithms_widebody(time_limit)

    def get_result_collector(self, aircraft_type):
        """根据机型获取对应的ResultCollector"""
        if AIRCRAFT_CONFIGS[aircraft_type]['type'] == 'narrowbody':
            from algorithm.for_narrow.base_algorithm1 import ResultCollector
        else:
            from algorithm.for_wide.base_algorithm import ResultCollector
        return ResultCollector()

    def _print_algorithm_result(self, algo_name, result, extra_info=''):
        """打印单个算法结果"""
        gap = result['evaluation']['cg_gap_percent']
        revenue = result['evaluation']['revenue']
        time_s = result['solve_time']
        mem = result['memory_peak_mb']
        feasible = '✓' if result['evaluation']['feasible'] else '✗'
        n_violations = len(result.get('violations', []))

        # 性能标记
        gap_mark = '★' if gap < 1 else ('☆' if gap < 10 else ' ')
        time_mark = '⚡' if time_s < 1 else ('⏱' if time_s < 10 else '🐢')

        print(f"        {algo_name:<8} | Gap: {gap:>8.2f}% {gap_mark} | Rev: {revenue:>8.0f} | "
              f"Time: {time_s:>6.2f}s {time_mark} | {feasible} | Viol: {n_violations} {extra_info}")

    def _print_eta(self, completed, total):
        """打印预计剩余时间"""
        if self.start_time and completed > 0:
            elapsed = time.time() - self.start_time
            eta = elapsed / completed * (total - completed)
            print(f"      [进度: {completed}/{total}, 已用: {elapsed / 60:.1f}min, 预计剩余: {eta / 60:.1f}min]")

    def run_multi_stage_tradeoff_experiment(self, aircraft_types=None, n_flights=10, time_limit=15):
        """
        实验1: Multi-Stage Trade-off Analysis (多阶段权衡分析)

        分析在多航段场景下，CG最小化与Revenue最大化之间的权衡
        使用5个代表性权重组合评估算法在不同优化目标下的表现
        """
        print("\n" + "=" * 80)
        print("实验1: Multi-Stage Trade-off Analysis (多阶段权衡分析)")
        print("=" * 80)

        print(f"配置: 航班数={n_flights}, 时间限制={time_limit}s")
        print(f"权重比例: {[(f'CG:{cg:.1f}, Rev:{rev:.1f}') for cg, rev in WEIGHT_RATIOS]}")
        print(f"约束: 舱位容量(1ULD/舱位) + ULD类型匹配 + 互斥舱位 + 重量限制")
        print("=" * 80)

        if aircraft_types is None:
            aircraft_types = list(AIRCRAFT_CONFIGS.keys())

        self.start_time = time.time()
        all_results = []
        intermediate_file = os.path.join(self.output_path, 'tradeoff_intermediate.csv')

        total_tasks = len(aircraft_types) * len(WEIGHT_RATIOS) * n_flights * 11
        completed_tasks = 0

        for ac_idx, aircraft_type in enumerate(aircraft_types):
            ac_class = AIRCRAFT_CONFIGS[aircraft_type]['type']
            is_widebody = (ac_class == 'widebody')

            print(f"\n{'=' * 80}")
            print(f"机型 [{ac_idx + 1}/{len(aircraft_types)}]: {aircraft_type} ({ac_class})")
            print('=' * 80)

            try:
                loader = MultiAircraftDataLoader(aircraft_type)
                loader.load_all()

                flights = loader.get_multi_flights(n_flights)
                if len(flights) == 0:
                    print(f"  无多航段航班，使用单航段")
                    flights = loader.get_single_flights(n_flights)

                print(f"  加载完成: {len(flights)} 个航班, {len(loader.cargo_holds)} 个舱位")

                algorithms = self.get_algorithms(aircraft_type, time_limit)

                for w_idx, (cg_w, rev_w) in enumerate(WEIGHT_RATIOS):
                    print(f"\n  ┌─ 权重比例 [{w_idx + 1}/{len(WEIGHT_RATIOS)}]: CG={cg_w:.1f}, Revenue={rev_w:.1f}")
                    print(f"  │")

                    collector = self.get_result_collector(aircraft_type)
                    weight_start = time.time()

                    for flight_idx, flight_num in enumerate(flights):
                        flight_cargo = loader.get_flight_cargo(flight_num)
                        if len(flight_cargo) == 0:
                            continue

                        # 使用带完整约束的Problem类
                        problem = CargoLoadingProblemMultiStage(
                            cargo_holds=loader.cargo_holds,
                            flight_params=loader.flight_params,
                            cg_limits=loader.cg_limits,
                            cargo_items=flight_cargo.reset_index(drop=True),
                            segment_type='multi',
                            cg_weight=cg_w,
                            revenue_weight=rev_w,
                            is_widebody=is_widebody
                        )

                        print(
                            f"  ├─ Flight [{flight_idx + 1}/{len(flights)}]: {flight_num} ({len(flight_cargo)} items)")

                        for algo_class in algorithms:
                            try:
                                algo = algo_class(problem, segment_type='multi', time_limit=time_limit)
                                result = algo.run_with_metrics()
                                result['aircraft_type'] = aircraft_type
                                result['cg_weight'] = cg_w
                                result['revenue_weight'] = rev_w
                                collector.add_result(result, flight_number=flight_num)

                                self._print_algorithm_result(algo.name, result)
                                completed_tasks += 1

                            except Exception as e:
                                print(f"        {algo_class.__name__:<8} | ERROR: {str(e)[:40]}")
                                completed_tasks += 1

                    # 汇总该权重点
                    weight_time = time.time() - weight_start
                    summary = collector.get_summary()

                    if summary:
                        print(f"  │")
                        print(f"  ├─ 汇总 (CG={cg_w:.1f}, Rev={rev_w:.1f}):")
                        print(f"  │  {'算法':<8} | {'平均Gap%':<12} | {'平均Revenue':<12} | {'平均时间':<10}")
                        print(f"  │  {'-' * 55}")

                        for algo, stats in summary.items():
                            algo_results = [r for r in collector.results if r['algorithm'] == algo]
                            avg_revenue = np.mean(
                                [r['evaluation']['revenue'] for r in algo_results]) if algo_results else 0

                            print(f"  │  {algo:<8} | {stats['avg_cg_gap_percent']:>10.2f}% | "
                                  f"{avg_revenue:>10.0f} | {stats['avg_solve_time']:>8.2f}s")

                            all_results.append({
                                'aircraft_type': aircraft_type,
                                'aircraft_class': ac_class,
                                'cg_weight': cg_w,
                                'revenue_weight': rev_w,
                                'algorithm': algo,
                                'avg_gap': stats['avg_cg_gap_percent'],
                                'std_gap': stats['std_cg_gap_percent'],
                                'avg_revenue': avg_revenue,
                                'avg_time': stats['avg_solve_time'],
                                'std_time': stats['std_solve_time'],
                                'avg_memory': stats['avg_memory_mb'],
                                'feasible_rate': stats['feasible_rate'],
                                'n_tests': stats['n_tests']
                            })

                    print(f"  └─ 耗时: {weight_time:.1f}s")
                    self._print_eta(completed_tasks, total_tasks)

                    # 实时保存
                    pd.DataFrame(all_results).to_csv(intermediate_file, index=False)

            except Exception as e:
                print(f"  错误: {e}")
                traceback.print_exc()

        # 保存最终结果
        total_time = time.time() - self.start_time
        results_df = pd.DataFrame(all_results)

        if len(results_df) > 0:
            results_file = os.path.join(self.output_path, 'tradeoff_results.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\n结果已保存: {results_file}")
            print(f"总耗时: {total_time / 60:.1f} 分钟")

            # 生成LaTeX表格
            self._generate_tradeoff_latex(results_df)

        return results_df

    def run_comparison_experiment(self, aircraft_types=None, n_pairs=15, time_limit=15):
        """
        实验2: Single-segment vs Multi-segment Comparison

        对比相同货物量下单航段和多航段的性能差异
        """
        print("\n" + "=" * 80)
        print("实验2: Single vs Multi-Segment Comparison (单/多航段对比)")
        print("=" * 80)

        print(f"配置: 配对数={n_pairs}, 时间限制={time_limit}s")
        print(f"约束: 舱位容量(1ULD/舱位) + ULD类型匹配 + 互斥舱位 + 重量限制")
        print("=" * 80)

        if aircraft_types is None:
            aircraft_types = list(AIRCRAFT_CONFIGS.keys())

        self.start_time = time.time()
        all_results = []
        intermediate_file = os.path.join(self.output_path, 'comparison_intermediate.csv')

        for ac_idx, aircraft_type in enumerate(aircraft_types):
            ac_class = AIRCRAFT_CONFIGS[aircraft_type]['type']
            is_widebody = (ac_class == 'widebody')

            print(f"\n{'=' * 80}")
            print(f"机型 [{ac_idx + 1}/{len(aircraft_types)}]: {aircraft_type} ({ac_class})")
            print('=' * 80)

            try:
                loader = MultiAircraftDataLoader(aircraft_type)
                loader.load_all()

                pairs = loader.get_paired_flights(n_pairs)
                if len(pairs) == 0:
                    print(f"  无配对航班")
                    continue

                print(f"  加载完成: {len(pairs)} 对航班")

                algorithms = self.get_algorithms(aircraft_type, time_limit)

                for pair_idx, pair in enumerate(pairs):
                    single_flight = pair['single_flight']
                    multi_flight = pair['multi_flight']

                    print(f"\n  ┌─ Pair [{pair_idx + 1}/{len(pairs)}]")
                    print(f"  │  Single: {single_flight} ({pair['single_weight']:.0f}kg)")
                    print(f"  │  Multi:  {multi_flight} ({pair['multi_weight']:.0f}kg)")

                    for segment_type, flight_num in [('single', single_flight), ('multi', multi_flight)]:
                        flight_cargo = loader.get_flight_cargo(flight_num)
                        if len(flight_cargo) == 0:
                            continue

                        # 单航段：CG优先 (cg_weight=1.0, revenue_weight=0.0)
                        # 多航段：Profit优先 (cg_weight=0.0, revenue_weight=1.0)
                        if segment_type == 'single':
                            cg_w, rev_w = 1.0, 0.0  # CG Priority
                        else:
                            cg_w, rev_w = 0.0, 1.0  # Profit Priority

                        problem = CargoLoadingProblemMultiStage(
                            cargo_holds=loader.cargo_holds,
                            flight_params=loader.flight_params,
                            cg_limits=loader.cg_limits,
                            cargo_items=flight_cargo.reset_index(drop=True),
                            segment_type=segment_type,
                            cg_weight=cg_w,
                            revenue_weight=rev_w,
                            is_widebody=is_widebody
                        )

                        print(f"  ├─ {segment_type.upper()} ({len(flight_cargo)} items) [cg_w={cg_w}, rev_w={rev_w}]:")

                        for algo_class in algorithms:
                            try:
                                algo = algo_class(problem, segment_type=segment_type, time_limit=time_limit)
                                result = algo.run_with_metrics()

                                self._print_algorithm_result(algo.name, result)

                                all_results.append({
                                    'aircraft_type': aircraft_type,
                                    'aircraft_class': ac_class,
                                    'pair_idx': pair_idx,
                                    'segment_type': segment_type,
                                    'flight_number': flight_num,
                                    'algorithm': algo.name,
                                    'cg_weight': cg_w,
                                    'revenue_weight': rev_w,
                                    'cg_gap': result['evaluation']['cg_gap_percent'],
                                    'revenue': result['evaluation']['revenue'],
                                    'solve_time': result['solve_time'],
                                    'memory_mb': result['memory_peak_mb'],
                                    'feasible': result['evaluation']['feasible'],
                                    'n_violations': len(result.get('violations', []))
                                })

                            except Exception as e:
                                print(f"        {algo_class.__name__:<8} | ERROR: {str(e)[:40]}")

                    # 实时保存
                    pd.DataFrame(all_results).to_csv(intermediate_file, index=False)

            except Exception as e:
                print(f"  错误: {e}")
                traceback.print_exc()

        # 保存最终结果
        total_time = time.time() - self.start_time
        results_df = pd.DataFrame(all_results)

        if len(results_df) > 0:
            results_file = os.path.join(self.output_path, 'comparison_results.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\n结果已保存: {results_file}")
            print(f"总耗时: {total_time / 60:.1f} 分钟")

            # 生成LaTeX表格
            self._generate_comparison_latex(results_df)

        return results_df

    def _generate_tradeoff_latex(self, results_df):
        """生成多阶段权衡分析的LaTeX表格"""
        if len(results_df) == 0:
            return

        for ac_class in ['narrowbody', 'widebody']:
            df = results_df[results_df['aircraft_class'] == ac_class]
            if len(df) == 0:
                continue

            aircraft_list = df['aircraft_type'].unique()

            latex = []
            latex.append(r"\begin{table*}[htp]")
            latex.append(r"\centering")
            latex.append(f"\\caption{{Multi-Stage Trade-off Analysis: {ac_class.title()} Aircraft}}")
            latex.append(f"\\label{{tab:tradeoff_{ac_class}}}")
            latex.append(r"\resizebox{\textwidth}{!}{")
            latex.append(r"\begin{tabular}{ll" + "cc" * len(aircraft_list) + "}")
            latex.append(r"\toprule")

            header = r"CG:Rev & Algorithm"
            for ac in aircraft_list:
                header += f" & \\multicolumn{{2}}{{c}}{{{ac}}}"
            header += r" \\"
            latex.append(header)

            subheader = " & "
            for _ in aircraft_list:
                subheader += " & Gap(\\%) & Rev"
            subheader += r" \\"
            latex.append(subheader)
            latex.append(r"\midrule")

            algos = ['MILP', 'QP', 'GA', 'PSO', 'MBO']

            for cg_w, rev_w in WEIGHT_RATIOS:
                for algo in algos:
                    row = f"{cg_w:.1f}:{rev_w:.1f} & {algo}"
                    for ac in aircraft_list:
                        data = df[(df['cg_weight'] == cg_w) &
                                  (df['algorithm'] == algo) &
                                  (df['aircraft_type'] == ac)]
                        if len(data) > 0:
                            gap = data['avg_gap'].values[0]
                            rev = data['avg_revenue'].values[0]
                            row += f" & {gap:.2f} & {rev:.0f}"
                        else:
                            row += " & - & -"
                    row += r" \\"
                    latex.append(row)
                latex.append(r"\midrule")

            latex.append(r"\bottomrule")
            latex.append(r"\end{tabular}}")
            latex.append(r"\end{table*}")

            latex_file = os.path.join(self.output_path, f'tradeoff_{ac_class}_table.tex')
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(latex))
            print(f"LaTeX表格已保存: {latex_file}")

    def _generate_comparison_latex(self, results_df):
        """生成单/多航段对比的LaTeX表格"""
        if len(results_df) == 0:
            return

        summary_data = []

        for aircraft_type in results_df['aircraft_type'].unique():
            ac_df = results_df[results_df['aircraft_type'] == aircraft_type]

            for algo in ac_df['algorithm'].unique():
                algo_df = ac_df[ac_df['algorithm'] == algo]

                single_df = algo_df[algo_df['segment_type'] == 'single']
                multi_df = algo_df[algo_df['segment_type'] == 'multi']

                if len(single_df) > 0 and len(multi_df) > 0:
                    summary_data.append({
                        'aircraft_type': aircraft_type,
                        'aircraft_class': AIRCRAFT_CONFIGS[aircraft_type]['type'],
                        'algorithm': algo,
                        'single_gap': single_df['cg_gap'].mean(),
                        'single_time': single_df['solve_time'].mean(),
                        'multi_gap': multi_df['cg_gap'].mean(),
                        'multi_time': multi_df['solve_time'].mean(),
                        'gap_overhead': multi_df['cg_gap'].mean() - single_df['cg_gap'].mean(),
                        'time_ratio': multi_df['solve_time'].mean() / single_df['solve_time'].mean() if single_df[
                                                                                                            'solve_time'].mean() > 0 else 0
                    })

        summary_df = pd.DataFrame(summary_data)

        for ac_class in ['narrowbody', 'widebody']:
            df = summary_df[summary_df['aircraft_class'] == ac_class]
            if len(df) == 0:
                continue

            aircraft_list = df['aircraft_type'].unique()

            latex = []
            latex.append(r"\begin{table*}[htp]")
            latex.append(r"\centering")
            latex.append(f"\\caption{{Single vs Multi-Segment Comparison: {ac_class.title()} Aircraft}}")
            latex.append(f"\\label{{tab:comparison_{ac_class}}}")
            latex.append(r"\resizebox{\textwidth}{!}{")
            latex.append(r"\begin{tabular}{l" + "cccc" * len(aircraft_list) + "}")
            latex.append(r"\toprule")

            header = r"Algorithm"
            for ac in aircraft_list:
                header += f" & \\multicolumn{{4}}{{c}}{{{ac}}}"
            header += r" \\"
            latex.append(header)

            subheader = ""
            for _ in aircraft_list:
                subheader += " & S-Gap & M-Gap & S-Time & M-Time"
            subheader += r" \\"
            latex.append(subheader)
            latex.append(r"\midrule")

            algos = ['MILP', 'MINLP', 'QP', 'DP', 'CP', 'GA', 'PSO', 'CS', 'ACO', 'ABC', 'MBO']

            for algo in algos:
                row = algo
                for ac in aircraft_list:
                    data = df[(df['algorithm'] == algo) & (df['aircraft_type'] == ac)]
                    if len(data) > 0:
                        row += f" & {data['single_gap'].values[0]:.2f}"
                        row += f" & {data['multi_gap'].values[0]:.2f}"
                        row += f" & {data['single_time'].values[0]:.2f}"
                        row += f" & {data['multi_time'].values[0]:.2f}"
                    else:
                        row += " & - & - & - & -"
                row += r" \\"
                latex.append(row)

            latex.append(r"\bottomrule")
            latex.append(r"\end{tabular}}")
            latex.append(r"\end{table*}")

            latex_file = os.path.join(self.output_path, f'comparison_{ac_class}_table.tex')
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(latex))
            print(f"LaTeX表格已保存: {latex_file}")

        summary_file = os.path.join(self.output_path, 'comparison_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"汇总结果已保存: {summary_file}")


def main():
    import argparse
    global BENCHMARK_PATH, CARGO_DATA_PATH, OUTPUT_PATH
    parser = argparse.ArgumentParser(description='Multi-Stage Optimization Tradeoff Experiment')
    parser.add_argument('--benchmark-path', type=str, default=BENCHMARK_PATH,
                        help='机型数据根目录(默认=BENCHMARK_PATH)')
    parser.add_argument('--cargo-data-path', type=str, default=CARGO_DATA_PATH,
                        help='货物/航班数据目录(默认=CARGO_DATA_PATH)')
    parser.add_argument('--output-path', type=str, default=OUTPUT_PATH,
                        help='输出目录(默认=OUTPUT_PATH)')

    parser.add_argument('--aircraft', type=str, nargs='+', default=None,
                        help='机型列表')
    parser.add_argument('--n-flights', type=int, default=10,
                        help='权衡实验的航班数 (默认10)')
    parser.add_argument('--time-limit', type=int, default=120,
                        help='算法时间限制 (默认15s)')

    args = parser.parse_args()

    # 允许命令行覆盖路径(不改变量名，只更新值)

    BENCHMARK_PATH = args.benchmark_path
    CARGO_DATA_PATH = args.cargo_data_path
    OUTPUT_PATH = args.output_path

    print("=" * 80)
    print("Multi-Stage Optimization Experiment")
    print("多阶段优化实验 - Tradeoff")
    print("=" * 80)
    print()
    print("路径配置:")
    print("  数据路径:", BENCHMARK_PATH)
    print("  输出路径:", OUTPUT_PATH)
    print()
    print("实验配置:")
    print("  机型:", args.aircraft or ['A320','B777'])
    print("  权衡实验航班数:", args.n_flights)
    print("  时间限制:", str(args.time_limit) + "s")
    print()
    print("机型分类:")
    print("  窄体机(algorithm1): A320")
    print("  宽体机(algorithm): B777")
    print()
    print("约束实现:")
    print("  ✓ 舱位容量约束 (每个舱位最多1个ULD) - 宽体机")
    print("  ✓ ULD类型匹配约束 - 宽体机")
    print("  ✓ 互斥舱位约束")
    print("  ✓ 重量约束")

    runner = MultiStageExperimentRunner()

    runner.run_multi_stage_tradeoff_experiment(
        args.aircraft,
        args.n_flights,
        args.time_limit
    )

    print()
    print("=" * 80)
    print("实验完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
