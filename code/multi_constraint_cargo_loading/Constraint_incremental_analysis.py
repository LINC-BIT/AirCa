#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ1 Experiment: Constraint Complexity Analysis
===============================================

Research Question: How does constraint complexity affect algorithm performance?

This experiment evaluates exact and heuristic algorithms under increasing
constraint complexity levels on the AirCa benchmark.

Two modes:
  - narrowbody: Multi-aircraft (A320, B777, etc.) with 11 algorithms,
                constraint levels 1/2/3 (Cargo Only / Cargo+Hold / Full)
  - widebody:   B777 with 5 exact + 6 heuristic algorithms,
                constraint levels loose/medium/tight

Project layout (expected):
    <code_root>/                              # e.g.  G:\\AirCa\\code
      algorithm/
        for_narrow/   (base_algorithm1.py, exact_algorithms1.py, ...)
        for_wide/     (base_algorithm.py,  exact_algorithms.py,  ...)
      multi_constraint_cargo_loading/
        Constraint_incremental_analysis.py    # <-- this file
      aircraft_data/
        A320/ B777/ C919/ ...

Usage:
  # Run from the multi_constraint_cargo_loading/ directory (auto-detects paths):
  python Constraint_incremental_analysis.py --mode both

  # Explicitly specify the code root (if auto-detection fails):
  python Constraint_incremental_analysis.py --code-path G:\\AirCa\\code

  # Run narrowbody only
  python Constraint_incremental_analysis.py --mode narrowbody

  # Run widebody only with 50 flights
  python Constraint_incremental_analysis.py --mode widebody --n-flights 50

  # Custom data paths
  python Constraint_incremental_analysis.py \\
      --benchmark-path /data/aircraft_data \\
      --cargo-data-path /data/bakFlightLoadData \\
      --output-path /data/output/rq1

  # Select specific aircraft types (narrowbody mode)
  python Constraint_incremental_analysis.py --mode narrowbody --aircraft A320
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from scipy import interpolate
import glob
import argparse

# =====================================================================
#  Auto-detect project root so that `from algorithm.xxx import ...` works
#  regardless of which subdirectory the script is launched from.
#
#  Expected layout:
#    <code_root>/
#      algorithm/
#        for_narrow/   (base_algorithm1.py, exact_algorithms1.py, ...)
#        for_wide/     (base_algorithm.py,  exact_algorithms.py,  ...)
#      multi_constraint_cargo_loading/
#        Constraint_incremental_analysis.py   <-- this file
#      aircraft_data/
#        A320/ B777/ ...
#
#  We walk upward from this file until we find a directory that contains
#  `algorithm/`, then insert it into sys.path.
# =====================================================================

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

# ==================== Aircraft Configurations ====================
AIRCRAFT_CONFIGS = {
    'B777': {'type': 'widebody', 'folder': 'B777', 'fleet_pattern': ['B777', '777', '77A', '77B', '77W']},
    'A320': {'type': 'narrowbody', 'folder': 'A320', 'fleet_pattern': ['A320', '320', '32A', '32B', '32N']},

}

# Narrowbody constraint level definitions
NARROWBODY_CONSTRAINT_LEVELS = {
    1: 'Cargo Only',       # Weight + CG only
    2: 'Cargo + Hold',     # + hold capacity, ULD type, exclusive holds
    3: 'Full Constraints',  # + loading order, continuous loading, DG isolation
}

# Widebody constraint level definitions
WIDEBODY_CONSTRAINT_LEVELS = ['loose', 'medium', 'tight']

# Timeout for single algorithm execution (seconds)
ALGO_TIMEOUT = 120


# =====================================================================
#  Utility
# =====================================================================

def run_with_timeout(func, timeout, *args, **kwargs):
    """Execute *func* in a daemon thread; return (result, error_string|None)."""
    result = [None]
    exception = [None]

    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, "TIMEOUT"
    if exception[0]:
        return None, str(exception[0])
    return result[0], None


# =====================================================================
#  Problem definitions
# =====================================================================

class NarrowbodyProblem:
    """Cargo-loading problem with 3 incremental constraint levels (narrowbody)."""

    def __init__(self, cargo_holds, flight_params, cg_limits, cargo_items,
                 segment_type='single', constraint_level=3):
        self.cargo_holds = cargo_holds
        self.flight_params = flight_params
        self.cg_limits = cg_limits
        self.cargo_items = cargo_items
        self.segment_type = segment_type
        self.constraint_level = constraint_level

        self.holds = self._get_unique_holds()
        self.n_holds = len(self.holds)
        self.n_items = len(cargo_items)

        self.initial_weight = flight_params['initial_weight']
        self.initial_cg = flight_params['initial_cg']

    def _get_unique_holds(self):
        holds, seen = [], set()
        for _, row in self.cargo_holds.iterrows():
            hid = row['hold_id']
            if hid not in seen:
                holds.append(row.to_dict())
                seen.add(hid)
        return holds

    def get_optimal_cg(self, total_weight):
        aft = self.cg_limits['aft_limit'](total_weight)
        fwd = self.cg_limits['fwd_limit'](total_weight)
        optimal = aft + (fwd - aft) * (1 / 3)
        return optimal, aft, fwd

    def calculate_cg(self, solution):
        total_moment = self.initial_cg
        total_weight = self.initial_weight
        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0:
                item = self.cargo_items.iloc[item_idx]
                hold = self.holds[hold_idx]
                total_weight += item['weight']
                total_moment += item['weight'] * hold['cg_coefficient']
        return total_moment, total_weight

    def evaluate_solution(self, solution):
        cg, total_weight = self.calculate_cg(solution)
        optimal_cg, aft_limit, fwd_limit = self.get_optimal_cg(total_weight)
        cg_gap = abs(cg - optimal_cg)
        cg_gap_percent = cg_gap / abs(optimal_cg) * 100 if optimal_cg != 0 else 0
        feasible = aft_limit <= cg <= fwd_limit

        revenue = 0
        if self.segment_type == 'multi':
            for idx, hidx in enumerate(solution):
                if hidx >= 0:
                    revenue += self.cargo_items.iloc[idx]['weight']

        return {
            'cg': cg, 'total_weight': total_weight, 'optimal_cg': optimal_cg,
            'cg_gap': cg_gap, 'cg_gap_percent': cg_gap_percent,
            'aft_limit': aft_limit, 'fwd_limit': fwd_limit,
            'feasible': feasible, 'revenue': revenue,
        }

    def check_constraints(self, solution):
        violations = []
        hold_weights = {i: 0 for i in range(self.n_holds)}
        hold_items = {i: [] for i in range(self.n_holds)}
        for item_idx, hold_idx in enumerate(solution):
            if hold_idx >= 0:
                hold_weights[hold_idx] += self.cargo_items.iloc[item_idx]['weight']
                hold_items[hold_idx].append(item_idx)

        # Level 1: weight
        for hidx, w in hold_weights.items():
            if w > self.holds[hidx]['max_weight']:
                violations.append({'type': 'weight', 'level': 1,
                                   'hold': self.holds[hidx]['hold_id'],
                                   'actual': w, 'limit': self.holds[hidx]['max_weight']})
        if self.constraint_level < 2:
            return violations

        # Level 2: exclusive holds + ULD type
        used = set(h for h in solution if h >= 0)
        for hidx in used:
            hold = self.holds[hidx]
            for exc in hold.get('exclusive_holds', []):
                if exc and exc.strip():
                    for oidx, oh in enumerate(self.holds):
                        if oh['hold_id'] == exc and oidx in used:
                            violations.append({'type': 'exclusive', 'level': 2,
                                               'hold1': hold['hold_id'], 'hold2': exc})

            uld_types = hold.get('uld_types', [])
            if uld_types:
                for iidx in hold_items.get(hidx, []):
                    ct = self.cargo_items.iloc[iidx].get('cargo_type', '')
                    if ct and ct not in uld_types:
                        violations.append({'type': 'uld_mismatch', 'level': 2,
                                           'hold': hold['hold_id']})
        if self.constraint_level < 3:
            return violations

        # Level 3: loading order, continuous loading, DG isolation
        for hidx in used:
            items_in = hold_items.get(hidx, [])
            if len(items_in) > 1:
                for i, i1 in enumerate(items_in):
                    for i2 in items_in[i + 1:]:
                        d1 = self.cargo_items.iloc[i1].get('destination', '')
                        d2 = self.cargo_items.iloc[i2].get('destination', '')
                        if d1 != d2:
                            violations.append({'type': 'loading_order', 'level': 3,
                                               'hold': self.holds[hidx]['hold_id']})

        sorted_used = sorted(used)
        for i in range(len(sorted_used) - 1):
            for mid in range(sorted_used[i] + 1, sorted_used[i + 1]):
                if mid not in used:
                    violations.append({'type': 'continuous_loading', 'level': 3,
                                       'gap_at': self.holds[mid]['hold_id']})

        dg_items = []
        for iidx, hidx in enumerate(solution):
            if hidx >= 0:
                ct = self.cargo_items.iloc[iidx].get('content_type', '')
                if ct == 'D' or 'DG' in str(ct).upper():
                    dg_items.append((iidx, hidx))
        for i, (_, h1) in enumerate(dg_items):
            for _, h2 in dg_items[i + 1:]:
                if abs(h1 - h2) < 2:
                    violations.append({'type': 'dangerous_isolation', 'level': 3})

        return violations


class WidebodyProblem:
    """Cargo-loading problem with loose/medium/tight constraint levels (widebody B777)."""

    def __init__(self, cargo_holds, flight_params, cg_limits, cargo_items,
                 segment_type='single', constraint_level='loose'):
        self.cargo_holds = cargo_holds
        self.flight_params = flight_params
        self.cg_limits = cg_limits
        self.cargo_items = cargo_items
        self.segment_type = segment_type
        self.constraint_level = constraint_level

        self.holds = self._get_unique_holds()
        self.n_holds = len(self.holds)
        self.n_items = len(cargo_items)
        self.initial_weight = flight_params['initial_weight']
        self.initial_cg = flight_params['initial_cg']

        self._setup_constraint_level()

    def _get_unique_holds(self):
        holds, seen = [], set()
        for _, row in self.cargo_holds.iterrows():
            hid = row['hold_id']
            if hid not in seen:
                holds.append(row.to_dict())
                seen.add(hid)
        return holds

    def _setup_constraint_level(self):
        if self.constraint_level == 'loose':
            for h in self.holds:
                h['original_max_weight'] = h['max_weight']
                h['max_weight'] = h['max_weight'] * 1.2
            self.cg_tolerance = 0.5
            self.check_exclusive = False
            self.check_capacity = False
            self.check_uld_type = False
        elif self.constraint_level == 'medium':
            for h in self.holds:
                h['original_max_weight'] = h['max_weight']
            self.cg_tolerance = 0.2
            self.check_exclusive = True
            self.check_capacity = True
            self.check_uld_type = True
        else:  # tight
            for h in self.holds:
                h['original_max_weight'] = h['max_weight']
                h['max_weight'] = h['max_weight'] * 0.8
            self.cg_tolerance = 0.0
            self.check_exclusive = True
            self.check_capacity = True
            self.check_uld_type = True

    def is_hold_compatible(self, item, hold):
        item_uld = ''
        if hasattr(item, 'uld_type'):
            item_uld = item.uld_type
        elif isinstance(item, dict):
            item_uld = item.get('uld_type', '')
        item_uld = str(item_uld).strip() if item_uld else ''
        allowed = hold.get('uld_types', [])
        if not allowed or not item_uld:
            return True
        return item_uld in allowed

    def get_optimal_cg(self, total_weight):
        aft = float(self.cg_limits['aft_limit'](total_weight))
        fwd = float(self.cg_limits['fwd_limit'](total_weight))
        if self.constraint_level == 'loose':
            margin = abs(fwd - aft) * 0.2
            aft -= margin; fwd += margin
        elif self.constraint_level == 'tight':
            margin = abs(fwd - aft) * 0.15
            aft += margin; fwd -= margin
        optimal = aft + (fwd - aft) * (1 / 3)
        return optimal, aft, fwd

    def calculate_cg(self, solution):
        total_moment = self.initial_cg
        total_weight = self.initial_weight
        for item_idx, hold_idx in enumerate(solution):
            if 0 <= hold_idx < len(self.holds):
                item = self.cargo_items.iloc[item_idx]
                hold = self.holds[hold_idx]
                cg_coef = hold['cg_coefficient']
                if cg_coef is not None:
                    total_weight += item['weight']
                    total_moment += item['weight'] * cg_coef
        return total_moment, total_weight

    def evaluate_solution(self, solution):
        cg, tw = self.calculate_cg(solution)
        optimal_cg, aft, fwd = self.get_optimal_cg(tw)
        cg_gap = abs(cg - optimal_cg)
        cg_gap_pct = cg_gap / abs(optimal_cg) * 100 if optimal_cg != 0 else 0

        if self.constraint_level == 'loose':
            cg_ok = True
        elif self.constraint_level == 'medium':
            cg_ok = (aft - self.cg_tolerance) <= cg <= (fwd + self.cg_tolerance)
        else:
            cg_ok = aft <= cg <= fwd

        violations = self.check_constraints(solution)
        feasible = cg_ok and len(violations) == 0

        revenue = 0
        if self.segment_type == 'multi':
            for idx, hidx in enumerate(solution):
                if hidx >= 0:
                    revenue += self.cargo_items.iloc[idx]['weight']

        return {
            'cg': cg, 'total_weight': tw, 'optimal_cg': optimal_cg,
            'cg_gap': cg_gap, 'cg_gap_percent': cg_gap_pct,
            'aft_limit': aft, 'fwd_limit': fwd,
            'feasible': feasible, 'revenue': revenue,
        }

    def check_constraints(self, solution):
        violations = []
        hold_weights = {i: 0 for i in range(self.n_holds)}
        hold_items = {i: [] for i in range(self.n_holds)}
        for iidx, hidx in enumerate(solution):
            if 0 <= hidx < self.n_holds:
                hold_weights[hidx] += self.cargo_items.iloc[iidx]['weight']
                hold_items[hidx].append(iidx)

        # Capacity (1 ULD per hold)
        if self.check_capacity:
            for hidx, ilist in hold_items.items():
                if len(ilist) > 1:
                    violations.append({'type': 'capacity', 'hold': self.holds[hidx]['hold_id'],
                                       'actual': len(ilist), 'limit': 1})

        # ULD type
        if self.check_uld_type:
            for iidx, hidx in enumerate(solution):
                if 0 <= hidx < self.n_holds:
                    if not self.is_hold_compatible(self.cargo_items.iloc[iidx], self.holds[hidx]):
                        violations.append({'type': 'uld_type', 'item': iidx,
                                           'hold': self.holds[hidx]['hold_id']})

        # Weight
        for hidx, w in hold_weights.items():
            if w > self.holds[hidx]['max_weight']:
                violations.append({'type': 'weight', 'hold': self.holds[hidx]['hold_id'],
                                   'actual': w, 'limit': self.holds[hidx]['max_weight']})

        # Exclusive
        if self.check_exclusive:
            checked = set()
            for hidx, hold in enumerate(self.holds):
                if hold_items[hidx] and hold.get('exclusive_holds'):
                    for exc_id in hold['exclusive_holds']:
                        for oidx, oh in enumerate(self.holds):
                            if oh['hold_id'] == exc_id and hold_items[oidx]:
                                pair = tuple(sorted([hidx, oidx]))
                                if pair not in checked:
                                    violations.append({'type': 'exclusive',
                                                       'hold1': hold['hold_id'], 'hold2': exc_id})
                                    checked.add(pair)

        # CG envelope (tight only)
        if self.constraint_level == 'tight':
            cg, tw = self.calculate_cg(solution)
            _, aft, fwd = self.get_optimal_cg(tw)
            if cg < aft or cg > fwd:
                violations.append({'type': 'cg_envelope', 'cg': cg,
                                   'aft_limit': aft, 'fwd_limit': fwd})

        return violations


# =====================================================================
#  Data loaders
# =====================================================================

class MultiAircraftDataLoader:
    """Data loader for the narrowbody experiment (multi-aircraft)."""

    def __init__(self, benchmark_path, cargo_data_path):
        self.benchmark_path = benchmark_path
        self.cargo_data_path = cargo_data_path
        self.cargo_data = None

    def load_aircraft_config(self, aircraft_type):
        config = AIRCRAFT_CONFIGS[aircraft_type]
        base = os.path.join(self.benchmark_path, config['folder'])
        return {
            'cargo_holds': self._load_cargo_holds(base, config['folder']),
            'flight_params': self._load_flight_params(base),
            'cg_limits': self._load_cg_limits(base),
            'aircraft_type': aircraft_type, 'config': config,
        }

    def _load_cargo_holds(self, base_path, folder):
        fp = os.path.join(base_path, f'{folder}.csv')
        if not os.path.exists(fp):
            for f in os.listdir(base_path):
                if f.endswith('.csv') and 'zfw' not in f.lower():
                    fp = os.path.join(base_path, f); break

        df = pd.read_csv(fp, header=None)
        holds = []
        for _, row in df.iterrows():
            hid = str(row[1]).strip() if pd.notna(row[1]) else None
            if hid is None or hid == 'nan':
                continue
            exc = str(row[2]).strip() if pd.notna(row[2]) else ''
            uld = str(row[3]).strip() if pd.notna(row[3]) else ''
            mw = float(row[4]) if pd.notna(row[4]) else 0

            cg_coef = None
            for i in range(len(row) - 1, -1, -1):
                if pd.notna(row[i]) and row[i] != '':
                    try: cg_coef = float(row[i]); break
                    except: continue

            arm = None
            for i in [8, 9, 10]:
                if i < len(row) and pd.notna(row[i]):
                    try: arm = float(row[i]); break
                    except: continue

            holds.append({
                'hold_id': hid,
                'exclusive_holds': [x for x in exc.split('/') if x],
                'uld_types': [x for x in uld.split('/') if x],
                'max_weight': mw, 'arm': arm, 'cg_coefficient': cg_coef,
            })
        return pd.DataFrame(holds)

    def _load_flight_params(self, base_path):
        fp_xlsx = os.path.join(base_path, '航班参数.xlsx')
        fp_csv = os.path.join(base_path, '航班参数.csv')
        if os.path.exists(fp_xlsx):
            df = pd.read_excel(fp_xlsx)
        elif os.path.exists(fp_csv):
            try:    df = pd.read_csv(fp_csv, encoding='gbk')
            except: df = pd.read_csv(fp_csv, encoding='utf-8')
        else:
            return {'initial_weight': 50000, 'initial_cg': 0}

        row = df.iloc[0]
        try:
            w1 = float(row.get('干操作重量', row.get('dryOperatingWeight', 50000)))
            c1 = float(row.get('干操作重心', row.get('dryOperatingCenter', 0)))
            w2 = float(row.get('旅客重量', row.get('passengerWeight', 10000)))
            c2 = float(row.get('旅客重心', row.get('passengerCenter', 0)))
        except:
            w1, c1, w2, c2 = 50000, 0, 10000, 0
        return {'initial_weight': w1 + w2, 'initial_cg': c1 + c2}

    def _load_cg_limits(self, base_path):
        af = os.path.join(base_path, 'stdZfw_a.csv')
        ff = os.path.join(base_path, 'stdZfw_f.csv')
        if os.path.exists(af) and os.path.exists(ff):
            da = pd.read_csv(af, header=None); df_ = pd.read_csv(ff, header=None)
            ai = interpolate.interp1d(da.iloc[:, 1].astype(float), da.iloc[:, 3].astype(float),
                                      kind='linear', fill_value='extrapolate')
            fi = interpolate.interp1d(df_.iloc[:, 1].astype(float), df_.iloc[:, 3].astype(float),
                                      kind='linear', fill_value='extrapolate')
        else:
            ai = lambda w: -0.1; fi = lambda w: 0.1
        return {'aft_limit': ai, 'fwd_limit': fi}

    def load_all_cargo_data(self):
        files = glob.glob(os.path.join(self.cargo_data_path, 'BAKFLGITH_LOADDATA*.csv'))
        if not files:
            raise FileNotFoundError(f"No cargo data found in {self.cargo_data_path}")
        parts = []
        for f in files:
            for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try: parts.append(pd.read_csv(f, encoding=enc, header=None)); break
                except: continue

        df = pd.concat(parts, ignore_index=True)
        df['flight_number'] = df.iloc[:, 0].astype(str)
        df['fleet'] = df.iloc[:, 1].astype(str).str.upper()
        df['destination'] = df.iloc[:, 2].astype(str).str.upper()
        df['weight'] = pd.to_numeric(df.iloc[:, 3], errors='coerce').fillna(0)
        df['content_type'] = df.iloc[:, 4].astype(str).str.upper()
        df['cargo_type'] = df.iloc[:, 7].astype(str) if df.shape[1] > 7 else ''
        df['volume'] = pd.to_numeric(df.iloc[:, 11], errors='coerce').fillna(0) if df.shape[1] > 11 else 0
        self.cargo_data = df
        return df

    def get_cargo_for_aircraft(self, aircraft_type, n_flights=100):
        if self.cargo_data is None:
            self.load_all_cargo_data()
        patterns = AIRCRAFT_CONFIGS[aircraft_type]['fleet_pattern']
        mask = self.cargo_data['fleet'].apply(lambda x: any(p in str(x).upper() for p in patterns))
        filtered = self.cargo_data[mask].copy()
        flights = filtered['flight_number'].unique()[:n_flights]
        return filtered[filtered['flight_number'].isin(flights)]


class B777DataLoader:
    """Data loader for the widebody B777 experiment."""

    def __init__(self, benchmark_path, cargo_data_path):
        self.base_path = os.path.join(benchmark_path, 'B777')
        self.cargo_data_path = cargo_data_path
        self.cargo_holds = None
        self.flight_params = None
        self.cg_limits = None
        self.cargo_data = None

    def load_cargo_holds(self):
        fp = os.path.join(self.base_path, 'B777.csv')
        if not os.path.exists(fp):
            for f in os.listdir(self.base_path):
                if f.endswith('.csv') and 'zfw' not in f.lower():
                    fp = os.path.join(self.base_path, f); break

        df = pd.read_csv(fp, header=None)
        holds = []
        for _, row in df.iterrows():
            hid = str(row[1]).strip() if pd.notna(row[1]) else None
            if hid is None or hid == 'nan':
                continue
            exc = str(row[2]).strip() if pd.notna(row[2]) else ''
            uld = str(row[3]).strip() if pd.notna(row[3]) else ''
            mw = float(row[4]) if pd.notna(row[4]) else 0

            cg_coef = None
            for i in range(len(row) - 1, -1, -1):
                if pd.notna(row[i]) and row[i] != '':
                    try: cg_coef = float(row[i]); break
                    except: continue

            arm = None
            for i in [8, 9, 10]:
                if i < len(row) and pd.notna(row[i]):
                    try: arm = float(row[i]); break
                    except: continue

            uld_list = [x.strip() for x in uld.split('/') if x.strip() and x.strip() != '////']
            holds.append({
                'hold_id': hid,
                'exclusive_holds': [x for x in exc.split('/') if x],
                'uld_types': uld_list, 'max_weight': mw,
                'arm': arm, 'cg_coefficient': cg_coef,
            })
        self.cargo_holds = pd.DataFrame(holds)
        print(f"  Loaded {len(self.cargo_holds)} cargo holds")
        return self.cargo_holds

    def load_flight_params(self):
        fp_csv = os.path.join(self.base_path, '航班参数.csv')
        fp_xlsx = os.path.join(self.base_path, '航班参数.xlsx')
        if os.path.exists(fp_csv):
            try:    df = pd.read_csv(fp_csv, encoding='gbk')
            except: df = pd.read_csv(fp_csv, encoding='utf-8')
        elif os.path.exists(fp_xlsx):
            df = pd.read_excel(fp_xlsx)
        else:
            self.flight_params = {'initial_weight': 150000, 'initial_cg': 0}
            return self.flight_params

        row = df.iloc[0]
        try:
            w1 = float(row.get('dryOperatingWeight', row.get('干操作重量', 150000)))
            c1 = float(row.get('dryOperatingCenter', row.get('干操作重心', 0)))
            w2 = float(row.get('passengerWeight', row.get('旅客重量', 20000)))
            c2 = float(row.get('passengerCenter', row.get('旅客重心', 0)))
        except:
            w1, c1, w2, c2 = 150000, 0, 20000, 0
        self.flight_params = {'initial_weight': w1 + w2, 'initial_cg': c1 + c2}
        return self.flight_params

    def load_cg_limits(self):
        af = os.path.join(self.base_path, 'stdZfw_a.csv')
        ff = os.path.join(self.base_path, 'stdZfw_f.csv')
        if os.path.exists(af) and os.path.exists(ff):
            da = pd.read_csv(af, header=None); df_ = pd.read_csv(ff, header=None)
            ai = interpolate.interp1d(da.iloc[:, 1].astype(float), da.iloc[:, 3].astype(float),
                                      kind='linear', fill_value='extrapolate')
            fi = interpolate.interp1d(df_.iloc[:, 1].astype(float), df_.iloc[:, 3].astype(float),
                                      kind='linear', fill_value='extrapolate')
        else:
            ai = lambda w: -0.1; fi = lambda w: 0.1
        self.cg_limits = {'aft_limit': ai, 'fwd_limit': fi}
        return self.cg_limits

    def load_cargo_data(self):
        files = glob.glob(os.path.join(self.cargo_data_path, 'BAKFLGITH_LOADDATA*.csv'))
        if not files:
            raise FileNotFoundError(f"No cargo data found in {self.cargo_data_path}")
        parts = []
        for f in files:
            for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try: parts.append(pd.read_csv(f, encoding=enc, header=None)); break
                except: continue

        df = pd.concat(parts, ignore_index=True)
        df['flight_number'] = df.iloc[:, 0].astype(str)
        df['fleet'] = df.iloc[:, 1].astype(str).str.upper()
        df['destination'] = df.iloc[:, 2].astype(str).str.upper()
        df['weight'] = pd.to_numeric(df.iloc[:, 3], errors='coerce').fillna(0)
        df['content_type'] = df.iloc[:, 4].astype(str).str.upper()
        if df.shape[1] > 7:
            df['uld_type'] = df.iloc[:, 7].astype(str).str.strip()
            df['uld_type'] = df['uld_type'].apply(lambda x: '' if x == 'nan' or pd.isna(x) else x)
        else:
            df['uld_type'] = ''
        df['volume'] = pd.to_numeric(df.iloc[:, 11], errors='coerce').fillna(0) if df.shape[1] > 11 else 0
        df['is_bulk'] = df['uld_type'].apply(lambda x: x == '' or x == 'nan' or pd.isna(x))

        def is_b777(s):
            s = str(s).upper().strip()
            return '777' in s or s.startswith('77') or s.startswith('B77')

        df = df[df['fleet'].apply(is_b777)].copy()
        dest_cnt = df.groupby('flight_number')['destination'].nunique()
        multi = set(dest_cnt[dest_cnt > 1].index)
        df['is_multi_segment'] = df['flight_number'].isin(multi)
        self.cargo_data = df
        print(f"  Loaded {len(df)} B777 cargo records")
        return self.cargo_data

    def get_top_flights(self, n_flights=100, single_only=True):
        if self.cargo_data is None:
            self.load_cargo_data()
        sub = self.cargo_data[~self.cargo_data['is_multi_segment']] if single_only else self.cargo_data
        counts = sub.groupby('flight_number').size().reset_index(name='count')
        counts = counts.sort_values('count', ascending=False)
        return counts['flight_number'].head(n_flights).tolist()

    def get_flight_cargo(self, flight_number):
        return self.cargo_data[self.cargo_data['flight_number'] == flight_number].copy()


# =====================================================================
#  Experiment runners
# =====================================================================

class NarrowbodyExperimentRunner:
    """Runner for the narrowbody constraint-level experiment."""

    def __init__(self, benchmark_path, cargo_data_path, output_path):
        self.loader = MultiAircraftDataLoader(benchmark_path, cargo_data_path)
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def get_algorithms(self):
        from algorithm.for_narrow.exact_algorithms1 import MILP, MINLP, QP, DP, CP
        from algorithm.for_narrow.heuristic_algorithms1 import GA, PSO, CS, ACO, ABC, MBO
        return {'exact': [MILP, MINLP, QP, DP, CP],
                'heuristic': [GA, PSO, CS, ACO, ABC, MBO]}

    def run(self, aircraft_types=None, n_flights=100, time_limit=30):
        from algorithm.for_narrow.base_algorithm1 import ResultCollector

        if aircraft_types is None:
            aircraft_types = list(AIRCRAFT_CONFIGS.keys())

        print("Loading cargo data ...")
        self.loader.load_all_cargo_data()
        algorithms = self.get_algorithms()
        all_results = []

        for ac in aircraft_types:
            print(f"\n{'=' * 60}\nAircraft: {ac}\n{'=' * 60}")
            try:
                config = self.loader.load_aircraft_config(ac)
                cargo = self.loader.get_cargo_for_aircraft(ac, n_flights)
                if len(cargo) == 0:
                    print("  Skipped: no cargo data"); continue

                flights = cargo['flight_number'].unique()
                print(f"  Flights: {len(flights)}, Cargo records: {len(cargo)}")

                for level in [1, 2, 3]:
                    print(f"\n  Constraint level {level}: {NARROWBODY_CONSTRAINT_LEVELS[level]}")
                    print(f"  {'-' * 55}")
                    collector = ResultCollector()
                    level_start = time.time()

                    for fi, fn in enumerate(flights[:n_flights]):
                        fc = cargo[cargo['flight_number'] == fn]
                        if len(fc) == 0: continue

                        n_items = len(fc)
                        print(f"\n    Flight [{fi+1}/{min(len(flights), n_flights)}]: {fn} ({n_items} items)")

                        problem = NarrowbodyProblem(
                            cargo_holds=config['cargo_holds'],
                            flight_params=config['flight_params'],
                            cg_limits=config['cg_limits'],
                            cargo_items=fc.reset_index(drop=True),
                            segment_type='single', constraint_level=level)

                        for algo_cls in algorithms['exact'] + algorithms['heuristic']:
                            algo_name = algo_cls.__name__
                            try:
                                algo = algo_cls(problem, segment_type='single', time_limit=time_limit)
                                result = algo.run_with_metrics()
                                result['aircraft_type'] = ac
                                result['constraint_level'] = level
                                collector.add_result(result, flight_number=fn)

                                gap = result['evaluation']['cg_gap_percent']
                                t = result['solve_time']
                                mem = result['memory_peak_mb']
                                feas = 'Y' if result['evaluation']['feasible'] else 'N'
                                print(f"      {algo_name:<8} | Gap: {gap:>8.2f}% | "
                                      f"Time: {t:>6.3f}s | Mem: {mem:>6.2f}MB | Feasible: {feas}")
                            except Exception as e:
                                print(f"      {algo_name:<8} | ERROR: {str(e)[:50]}")

                    # Per-level summary
                    level_elapsed = time.time() - level_start
                    summary = collector.get_summary()
                    if summary:
                        print(f"\n  ┌─ Level {level} Summary ({level_elapsed:.1f}s):")
                        print(f"  │  {'Algorithm':<8} | {'AvgGap%':>10} | {'AvgTime':>8} | {'Feasible%':>9} | {'Count':>5}")
                        print(f"  │  {'-' * 55}")
                        for algo, stats in summary.items():
                            all_results.append({
                                'aircraft_type': ac, 'constraint_level': level,
                                'constraint_name': NARROWBODY_CONSTRAINT_LEVELS[level],
                                'algorithm': algo,
                                'avg_gap': stats['avg_cg_gap_percent'],
                                'std_gap': stats['std_cg_gap_percent'],
                                'avg_time': stats['avg_solve_time'],
                                'std_time': stats['std_solve_time'],
                                'avg_memory': stats['avg_memory_mb'],
                                'std_memory': stats['std_memory_mb'],
                                'feasible_rate': stats['feasible_rate'],
                                'n_tests': stats['n_tests'],
                            })
                            print(f"  │  {algo:<8} | {stats['avg_cg_gap_percent']:>9.2f}% | "
                                  f"{stats['avg_solve_time']:>7.3f}s | {stats['feasible_rate']:>8.1f}% | "
                                  f"{stats['n_tests']:>5}")
                        print(f"  └─ Total records: {len(collector.results)}")
            except Exception as e:
                print(f"  Error: {e}"); traceback.print_exc()

        df = pd.DataFrame(all_results)
        out = os.path.join(self.output_path, 'narrowbody_constraint_results.csv')
        df.to_csv(out, index=False)
        print(f"\nResults saved: {out}")
        self._generate_latex(df)
        return df

    def _generate_latex(self, df):
        for level in [1, 2, 3]:
            sub = df[df['constraint_level'] == level]
            lines = [r"\begin{table*}[htp]", r"\centering",
                     f"\\caption{{Constraint Level {level}: {NARROWBODY_CONSTRAINT_LEVELS[level]}}}",
                     f"\\label{{tab:constraint_level_{level}}}",
                     r"\resizebox{\textwidth}{!}{",
                     r"\begin{tabular}{l" + "c" * len(AIRCRAFT_CONFIGS) + "}", r"\toprule"]
            hdr = "Algorithm"
            for ac in AIRCRAFT_CONFIGS: hdr += f" & {ac}"
            lines.append(hdr + r" \\"); lines.append(r"\midrule")

            for algo in ['MILP','MINLP','QP','DP','CP','GA','PSO','CS','ACO','ABC','MBO']:
                row = algo
                for ac in AIRCRAFT_CONFIGS:
                    d = sub[(sub['algorithm'] == algo) & (sub['aircraft_type'] == ac)]
                    if len(d) > 0:
                        row += f" & ${d['avg_gap'].values[0]:.2f} \\scriptstyle \\pm {d['std_gap'].values[0]:.2f}$"
                    else:
                        row += " & -"
                lines.append(row + r" \\")

            lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
            fp = os.path.join(self.output_path, f'narrowbody_level_{level}.tex')
            with open(fp, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"LaTeX table saved: {fp}")


class WidebodyExperimentRunner:
    """Runner for the widebody B777 constraint-level experiment."""

    def __init__(self, benchmark_path, cargo_data_path, output_path):
        self.loader = B777DataLoader(benchmark_path, cargo_data_path)
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.all_results = []

    def get_exact_algorithms(self):
        from algorithm.for_wide.exact_algorithms import MILP, MINLP, QP, DP, CP
        from algorithm.for_wide.heuristic_algorithms import GA, PSO, CS, ACO, ABC, MBO
        return [MILP, MINLP, QP, DP, CP, GA, PSO, CS, ACO, ABC, MBO]

    def _run_single(self, algo_cls, problem, time_limit):
        try:
            algo = algo_cls(problem, segment_type='single', time_limit=time_limit)
            return algo.run_with_metrics(), 'OK'
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def run(self, n_flights=20, time_limit=30):
        print("\nLoading B777 data ...")
        self.loader.load_cargo_holds()
        self.loader.load_flight_params()
        self.loader.load_cg_limits()
        self.loader.load_cargo_data()

        flights = self.loader.get_top_flights(n_flights)
        print(f"  Flights: {len(flights)}, Holds: {len(self.loader.cargo_holds)}")
        if not flights:
            print("WARNING: no flights found!"); return pd.DataFrame()

        algos = self.get_exact_algorithms()
        print(f"  Algorithms: {[a.__name__ for a in algos]}")
        self.all_results = []
        start = time.time()

        for level in WIDEBODY_CONSTRAINT_LEVELS:
            print(f"\n{'=' * 80}\nConstraint level: {level.upper()}\n{'=' * 80}")
            lstart = time.time()

            for fi, fn in enumerate(flights):
                fc = self.loader.get_flight_cargo(fn)
                if len(fc) == 0: continue
                print(f"  Flight [{fi+1}/{len(flights)}]: {fn} ({len(fc)} items)")

                try:
                    problem = WidebodyProblem(
                        cargo_holds=self.loader.cargo_holds,
                        flight_params=self.loader.flight_params,
                        cg_limits=self.loader.cg_limits,
                        cargo_items=fc.reset_index(drop=True),
                        segment_type='single', constraint_level=level)
                except Exception as e:
                    print(f"    Problem creation failed: {e}"); continue

                for acls in algos:
                    aname = acls.__name__
                    result, status = run_with_timeout(self._run_single, ALGO_TIMEOUT,
                                                     acls, problem, time_limit)
                    rec = {'constraint_level': level, 'flight_number': fn,
                           'algorithm': aname, 'n_items': len(fc),
                           'cg_gap_percent': None, 'solve_time': None,
                           'memory_mb': None, 'feasible': None,
                           'n_violations': None, 'status': 'UNKNOWN'}

                    if result is None:
                        rec['status'] = 'TIMEOUT' if status == 'TIMEOUT' else f'ERROR: {status}'
                        if status == 'TIMEOUT':
                            rec['solve_time'] = ALGO_TIMEOUT
                            print(f"    {aname:<8} | TIMEOUT ({ALGO_TIMEOUT}s)")
                        else:
                            print(f"    {aname:<8} | ERROR: {str(status)[:50]}")
                    else:
                        inner, istatus = result
                        if inner is not None and istatus == 'OK':
                            rec.update({
                                'cg_gap_percent': inner['evaluation']['cg_gap_percent'],
                                'solve_time': inner['solve_time'],
                                'memory_mb': inner['memory_peak_mb'],
                                'feasible': inner['evaluation']['feasible'],
                                'n_violations': len(inner.get('violations', [])),
                                'status': 'OK',
                            })
                            feas = 'Y' if rec['feasible'] else 'N'
                            print(f"    {aname:<8} | Gap: {rec['cg_gap_percent']:>8.2f}% | "
                                  f"Time: {rec['solve_time']:>6.3f}s | Mem: {rec['memory_mb']:>6.2f}MB ")
                        else:
                            rec['status'] = f'ERROR: {istatus}'
                            print(f"    {aname:<8} | ERROR: {str(istatus)[:50]}")

                    self.all_results.append(rec)

            elapsed = time.time() - lstart
            ok = [r for r in self.all_results if r['constraint_level'] == level and r['status'] == 'OK']
            if ok:
                okdf = pd.DataFrame(ok)
                print(f"\n  Summary ({level}): {len(ok)} results in {elapsed:.1f}s")
                for a in okdf['algorithm'].unique():
                    sub = okdf[okdf['algorithm'] == a]
                    print(f"    {a:<8} AvgGap:{sub['cg_gap_percent'].mean():>8.2f}%  "
                          f"AvgTime:{sub['solve_time'].mean():>6.2f}s")

        df = pd.DataFrame(self.all_results)
        out = os.path.join(self.output_path, 'widebody_constraint_results.csv')
        df.to_csv(out, index=False)
        print(f"\nResults saved: {out}")
        print(f"Total time: {(time.time() - start) / 60:.1f} min")
        self._generate_latex(df)
        return df

    def _generate_latex(self, df):
        ok = df[df['status'] == 'OK']
        if len(ok) == 0: return
        lines = [r"\begin{table*}[htp]", r"\centering",
                 r"\caption{Exact Algorithm Performance under Different Constraint Levels (B777)}",
                 r"\label{tab:constraint_levels_b777}",
                 r"\begin{tabular}{l|ccc|ccc|ccc}", r"\toprule",
                 r" & \multicolumn{3}{c|}{Gap (\%)} & \multicolumn{3}{c|}{Time (s)} & \multicolumn{3}{c}{Feasible Rate (\%)} \\",
                 r"Algorithm & Loose & Medium & Tight & Loose & Medium & Tight & Loose & Medium & Tight \\",
                 r"\midrule"]
        for algo in ['MILP','MINLP','QP','DP','CP']:
            row = algo
            for metric in ['cg_gap_percent', 'solve_time', 'feasible']:
                for lv in WIDEBODY_CONSTRAINT_LEVELS:
                    d = ok[(ok['algorithm'] == algo) & (ok['constraint_level'] == lv)]
                    if len(d) > 0:
                        v = d[metric].mean() * (100 if metric == 'feasible' else 1)
                        row += f" & {v:.2f}" if metric != 'feasible' else f" & {v:.1f}"
                    else:
                        row += " & -"
            lines.append(row + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        fp = os.path.join(self.output_path, 'widebody_constraint_table.tex')
        with open(fp, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"LaTeX table saved: {fp}")


# =====================================================================
#  CLI
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description='RQ1: Constraint Complexity Analysis on AirCa Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect paths (run from multi_constraint_cargo_loading/):
  python Constraint_incremental_analysis.py --mode both

  # Specify code root if auto-detection fails:
  python Constraint_incremental_analysis.py --code-path G:\\AirCa\\code

  # Narrowbody only, specific aircraft:
  python Constraint_incremental_analysis.py --mode narrowbody --aircraft A320

  # Widebody with more flights:
  python Constraint_incremental_analysis.py --mode widebody --n-flights 50
        """)

    parser.add_argument('--code-path', type=str, default=None,
                        help="Path to the directory containing 'algorithm/' package. "
                             "Auto-detected if not specified.")
    parser.add_argument('--mode', type=str, choices=['narrowbody', 'widebody', 'both'],
                        default='both',
                        help='Experiment mode (default: both)')
    parser.add_argument('--benchmark-path', type=str,
                        default=r'G:\AirCa\code\aircraft_data',
                        help='Root path of the AirCa benchmark dataset')
    parser.add_argument('--cargo-data-path', type=str,
                        default=r'G:\loading_benchmark\bakFlightLoadData\bakFlightLoadData',
                        help='Path to cargo flight load data')
    parser.add_argument('--output-path', type=str,
                        default=r'G:\AirCa\code\AirCa_output\constraint_analysis3',
                        help='Directory for experiment outputs')
    parser.add_argument('--aircraft', type=str, nargs='+', default=None,
                        help='Aircraft types to test (narrowbody mode, default: all)')
    parser.add_argument('--n-flights', type=int, default=1,
                        help='Number of flights per aircraft type (default: 10)')
    parser.add_argument('--time-limit', type=int, default=30,
                        help='Per-algorithm time limit in seconds (default: 30)')
    parser.add_argument('--algo-timeout', type=int, default=120,
                        help='Hard timeout per algorithm call in seconds (default: 120)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    global ALGO_TIMEOUT
    ALGO_TIMEOUT = args.algo_timeout

    # Handle --code-path: explicitly add to sys.path for algorithm imports
    if args.code_path:
        code_path = os.path.abspath(args.code_path)
        if os.path.isdir(os.path.join(code_path, 'algorithm')):
            if code_path not in sys.path:
                sys.path.insert(0, code_path)
                print(f"[path] Added '{code_path}' to sys.path via --code-path")
        else:
            print(f"[WARNING] --code-path '{code_path}' does not contain 'algorithm/' directory")

    print("=" * 70)
    print("RQ1: Constraint Complexity Analysis")
    print("=" * 70)
    print(f"\n  Mode:            {args.mode}")
    print(f"  Benchmark path:  {args.benchmark_path}")
    print(f"  Cargo data path: {args.cargo_data_path}")
    print(f"  Output path:     {args.output_path}")
    print(f"  Flights:         {args.n_flights}")
    print(f"  Time limit:      {args.time_limit}s")
    print(f"  Algo timeout:    {args.algo_timeout}s")

    if args.mode in ('narrowbody', 'both'):
        print("\n" + "=" * 70)
        print("  Narrowbody Experiment")
        print("  Constraint levels: 1=Cargo Only, 2=Cargo+Hold, 3=Full")
        print("=" * 70)

        runner = NarrowbodyExperimentRunner(
            args.benchmark_path, args.cargo_data_path, args.output_path)
        runner.run(aircraft_types=args.aircraft,
                   n_flights=args.n_flights, time_limit=args.time_limit)

    if args.mode in ('widebody', 'both'):
        print("\n" + "=" * 70)
        print("  Widebody B777 Experiment")
        print("  Constraint levels: loose / medium / tight")
        print("=" * 70)

        runner = WidebodyExperimentRunner(
            args.benchmark_path, args.cargo_data_path, args.output_path)
        runner.run(n_flights=args.n_flights, time_limit=args.time_limit)

    print("\n" + "=" * 70)
    print("Experiment finished!")
    print("=" * 70)


if __name__ == '__main__':
    main()
