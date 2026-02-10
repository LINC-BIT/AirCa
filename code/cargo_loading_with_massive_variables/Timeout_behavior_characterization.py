#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A320 Experiments (Split from scaling_cargo_exp.py)

- Experiment 1: Variable Scaling Analysis
- Experiment 2: Timeout Behavior Characterization

Changes vs original scaling_cargo_exp.py:
1) Only A320 data (narrow-body).
2) All hard-coded paths are CLI configurable (with original paths as defaults).
3) Algorithm imports match your current structure:
   algorithm/for_narrow/{base_algorithm1.py, exact_algorithms1.py, heuristic_algorithms1.py}
   (NO milp_algorithm1.py). Robust fallback import is included.
4) Adds explicit constraint checks (weight + exclusive + optional uld_type + optional cg envelope),
   and prints violation counts.

Notes:
- This script expects your project code root contains the folder 'algorithm'.
- Aircraft data is expected under --aircraft-data-dir (default G:\AirCa\code\aircraft_data).
  The loader searches for A320.csv and other required files in several common layouts.

Python: 3.8+
"""

import argparse


import os
import sys
import glob
import time
import threading
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from scipy import interpolate

# --------------------- timeout helper ---------------------

def run_with_timeout(func, timeout, *args, **kwargs):
    """Run func(*args, **kwargs) in a thread and enforce timeout (seconds)."""
    result = [None]
    exception = [None]

    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, "TIMEOUT"
    if exception[0] is not None:
        return None, f"{type(exception[0]).__name__}: {exception[0]}"
    return result[0], None

# --------------------- robust imports ---------------------

def _dynamic_import(module_name: str, file_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def collect_algorithm_classes(exact_mod, heu_mod) -> List[Type]:
    """Collect algorithm classes in a stable order; fall back to scanning."""
    preferred = ["MILP", "MINLP", "QP", "DP", "CP", "GA", "PSO", "CS", "ACO", "ABC", "MBO"]
    algo_classes: List[Type] = []
    for name in preferred:
        cls = getattr(exact_mod, name, None) or getattr(heu_mod, name, None)
        if isinstance(cls, type):
            algo_classes.append(cls)

    # fallback: scan for classes with run_with_metrics
    if not algo_classes:
        candidates = []
        for mod in (exact_mod, heu_mod):
            for _, v in vars(mod).items():
                if isinstance(v, type) and hasattr(v, "run_with_metrics"):
                    candidates.append(v)
        seen = set()
        for c in candidates:
            if c.__name__ not in seen:
                algo_classes.append(c)
                seen.add(c.__name__)
    return algo_classes

def import_narrow_algorithms(code_root: str):
    """
    Expected structure:
      <code_root>/
        algorithm/
          for_narrow/
            base_algorithm1.py
            exact_algorithms1.py
            heuristic_algorithms1.py
    Returns:
      CargoLoadingProblem (or None), ResultCollector (or None), algo_classes (list), import_mode (str)
    """
    if code_root and code_root not in sys.path:
        sys.path.insert(0, code_root)

    narrow_dir = os.path.join(code_root, "algorithm", "for_narrow")

    # 1) package import
    try:
        import importlib
        base = importlib.import_module("algorithm.for_narrow.base_algorithm1")
        exact_mod = importlib.import_module("algorithm.for_narrow.exact_algorithms1")
        heu_mod = importlib.import_module("algorithm.for_narrow.heuristic_algorithms1")

        CargoLoadingProblem = getattr(base, "CargoLoadingProblem", None)
        ResultCollector = getattr(base, "ResultCollector", None)

        algo_classes = collect_algorithm_classes(exact_mod, heu_mod)
        if not algo_classes:
            raise ImportError("No algorithm classes found via package import.")
        return CargoLoadingProblem, ResultCollector, algo_classes, "algorithm.for_narrow (package)"
    except Exception:
        pass

    # 2) flat import
    try:
        if os.path.isdir(narrow_dir) and narrow_dir not in sys.path:
            sys.path.insert(0, narrow_dir)

        import importlib
        base = importlib.import_module("base_algorithm1")
        exact_mod = importlib.import_module("exact_algorithms1")
        heu_mod = importlib.import_module("heuristic_algorithms1")

        CargoLoadingProblem = getattr(base, "CargoLoadingProblem", None)
        ResultCollector = getattr(base, "ResultCollector", None)

        algo_classes = collect_algorithm_classes(exact_mod, heu_mod)
        if not algo_classes:
            raise ImportError("No algorithm classes found via flat import.")
        return CargoLoadingProblem, ResultCollector, algo_classes, "for_narrow folder (flat)"
    except Exception:
        pass

    # 3) dynamic import
    base = _dynamic_import("base_algorithm1_dyn", os.path.join(narrow_dir, "base_algorithm1.py"))
    exact_mod = _dynamic_import("exact_algorithms1_dyn", os.path.join(narrow_dir, "exact_algorithms1.py"))
    heu_mod = _dynamic_import("heuristic_algorithms1_dyn", os.path.join(narrow_dir, "heuristic_algorithms1.py"))

    CargoLoadingProblem = getattr(base, "CargoLoadingProblem", None)
    ResultCollector = getattr(base, "ResultCollector", None)

    algo_classes = collect_algorithm_classes(exact_mod, heu_mod)
    if not algo_classes:
        raise ImportError("No algorithm classes found via dynamic import.")
    return CargoLoadingProblem, ResultCollector, algo_classes, "for_narrow folder (dynamic)"

# --------------------- local fallback problem ---------------------

class LocalCargoLoadingProblem:
    """
    Fallback CargoLoadingProblem with:
      - CG evaluation
      - constraint checks:
          * weight (per hold max_weight)
          * exclusive holds (if enabled)
          * optional uld_type compatibility (if cargo has 'uld_type' or 'cargo_type')
          * optional cg envelope (if constraint_level == 'tight')
    """

    def __init__(
        self,
        cargo_holds: pd.DataFrame,
        flight_params: Dict[str, float],
        cg_limits: Dict[str, Any],
        cargo_items: pd.DataFrame,
        segment_type: str = "single",
        check_exclusive: bool = True,
        constraint_level: str = "basic",
    ):
        self.cargo_holds = cargo_holds
        self.flight_params = flight_params
        self.cg_limits = cg_limits
        self.cargo_items = cargo_items
        self.segment_type = segment_type

        self.check_exclusive = check_exclusive
        self.constraint_level = constraint_level  # 'basic' or 'tight'

        self.holds = self._get_unique_holds()
        self.n_holds = len(self.holds)
        self.n_items = len(cargo_items)

        self.initial_weight = float(flight_params.get("initial_weight", 50000))
        self.initial_cg = float(flight_params.get("initial_cg", 0))

    def _get_unique_holds(self):
        holds = []
        seen = set()
        for _, row in self.cargo_holds.iterrows():
            hold_id = row["hold_id"]
            if hold_id not in seen:
                holds.append(row.to_dict())
                seen.add(hold_id)
        return holds

    def get_optimal_cg(self, total_weight):
        aft = float(self.cg_limits["aft_limit"](total_weight))
        fwd = float(self.cg_limits["fwd_limit"](total_weight))
        optimal = aft + (fwd - aft) * (1 / 3)
        return optimal, aft, fwd

    def calculate_cg(self, solution):
        total_moment = self.initial_cg
        total_weight = self.initial_weight

        for item_idx, hold_idx in enumerate(solution):
            if hold_idx is None:
                continue
            if isinstance(hold_idx, (np.integer, int)) and 0 <= int(hold_idx) < len(self.holds):
                item = self.cargo_items.iloc[item_idx]
                hold = self.holds[int(hold_idx)]
                weight = float(item.get("weight", 0))
                cg_coef = hold.get("cg_coefficient", None)
                if cg_coef is not None:
                    total_weight += weight
                    total_moment += weight * float(cg_coef)

        return total_moment, total_weight

    def evaluate_solution(self, solution):
        cg, total_weight = self.calculate_cg(solution)
        optimal_cg, aft_limit, fwd_limit = self.get_optimal_cg(total_weight)

        cg_gap = abs(cg - optimal_cg)
        cg_gap_percent = cg_gap / abs(optimal_cg) * 100 if optimal_cg != 0 else 0
        feasible = aft_limit <= cg <= fwd_limit

        revenue = 0.0
        if self.segment_type == "multi":
            for item_idx, hold_idx in enumerate(solution):
                if hold_idx is not None and int(hold_idx) >= 0:
                    revenue += float(self.cargo_items.iloc[item_idx].get("weight", 0))

        return {
            "cg": cg,
            "total_weight": total_weight,
            "optimal_cg": optimal_cg,
            "cg_gap": cg_gap,
            "cg_gap_percent": cg_gap_percent,
            "aft_limit": aft_limit,
            "fwd_limit": fwd_limit,
            "feasible": feasible,
            "revenue": revenue,
        }

    def is_hold_compatible(self, item_row: pd.Series, hold: Dict[str, Any]) -> bool:
        hold_uld_types = hold.get("uld_types", []) or []
        if not hold_uld_types:
            return True

        uld = None
        if "uld_type" in item_row:
            uld = str(item_row.get("uld_type", "")).strip()
        elif "cargo_type" in item_row:
            uld = str(item_row.get("cargo_type", "")).strip()
        if not uld:
            return True

        return any(uld.upper() == str(t).upper() for t in hold_uld_types)

    def check_constraints(self, solution):
        violations = []

        hold_weights = {i: 0.0 for i in range(self.n_holds)}
        hold_items = {i: [] for i in range(self.n_holds)}

        for item_idx, hold_idx in enumerate(solution):
            if hold_idx is None:
                continue
            try:
                h = int(hold_idx)
            except Exception:
                continue
            if 0 <= h < self.n_holds:
                item = self.cargo_items.iloc[item_idx]
                hold_weights[h] += float(item.get("weight", 0))
                hold_items[h].append(int(item_idx))

                if not self.is_hold_compatible(item, self.holds[h]):
                    violations.append({"type": "uld_type", "item": int(item_idx), "hold": self.holds[h]["hold_id"]})

        # Weight
        for hidx, w in hold_weights.items():
            limit = float(self.holds[hidx].get("max_weight", 0) or 0)
            if limit > 0 and w > limit + 1e-6:
                violations.append({"type": "weight", "hold": self.holds[hidx]["hold_id"], "actual": w, "limit": limit})

        # Exclusive holds
        if self.check_exclusive:
            checked = set()
            used = {h for h, arr in hold_items.items() if arr}
            for hidx in used:
                hold = self.holds[hidx]
                for exc_id in hold.get("exclusive_holds", []) or []:
                    if not exc_id:
                        continue
                    for oidx, oh in enumerate(self.holds):
                        if oh["hold_id"] == exc_id and oidx in used:
                            pair = tuple(sorted([hidx, oidx]))
                            if pair not in checked:
                                violations.append({"type": "exclusive", "hold1": hold["hold_id"], "hold2": exc_id})
                                checked.add(pair)

        # CG envelope (tight)
        if self.constraint_level == "tight":
            cg, tw = self.calculate_cg(solution)
            _, aft, fwd = self.get_optimal_cg(tw)
            if cg < aft or cg > fwd:
                violations.append({"type": "cg_envelope", "cg": cg, "aft_limit": aft, "fwd_limit": fwd})

        return violations

# --------------------- data loader (A320 only) ---------------------

class A320DataLoader:
    def __init__(self, aircraft_data_dir: str, cargo_data_dir: str, aircraft_type: str = "A320"):
        self.aircraft_type = aircraft_type
        self.aircraft_data_dir = aircraft_data_dir
        self.cargo_data_dir = cargo_data_dir

        self.cargo_holds: Optional[pd.DataFrame] = None
        self.flight_params: Optional[Dict[str, float]] = None
        self.cg_limits: Optional[Dict[str, Any]] = None
        self.cargo_data: Optional[pd.DataFrame] = None

    def _candidate_aircraft_dirs(self) -> List[str]:
        cands = [self.aircraft_data_dir]
        nested = os.path.join(self.aircraft_data_dir, self.aircraft_type)
        if os.path.isdir(nested):
            cands.append(nested)
        return cands

    def load_cargo_holds(self) -> pd.DataFrame:
        filepath = None
        for d in self._candidate_aircraft_dirs():
            p = os.path.join(d, f"{self.aircraft_type}.csv")
            if os.path.exists(p):
                filepath = p
                break

        if filepath is None:
            for d in self._candidate_aircraft_dirs():
                for p in glob.glob(os.path.join(d, "*.csv")):
                    if self.aircraft_type.upper() in os.path.basename(p).upper() and "ZFW" not in p.upper():
                        filepath = p
                        break
                if filepath:
                    break

        if filepath is None:
            raise FileNotFoundError(f"Cannot find {self.aircraft_type}.csv under {self.aircraft_data_dir}")

        df = pd.read_csv(filepath, header=None)

        cargo_holds = []
        for _, row in df.iterrows():
            hold_id = str(row[1]).strip() if len(row) > 1 and pd.notna(row[1]) else None
            if not hold_id or hold_id.lower() == "nan":
                continue

            exclusive = str(row[2]).strip() if len(row) > 2 and pd.notna(row[2]) else ""
            uld_types = str(row[3]).strip() if len(row) > 3 and pd.notna(row[3]) else ""
            max_weight = float(row[4]) if len(row) > 4 and pd.notna(row[4]) else 0.0

            cg_coef = None
            for i in range(len(row) - 1, -1, -1):
                if pd.notna(row[i]) and row[i] != "":
                    try:
                        cg_coef = float(row[i])
                        break
                    except Exception:
                        continue

            arm = None
            for i in (8, 9, 10):
                if i < len(row) and pd.notna(row[i]):
                    try:
                        arm = float(row[i])
                        break
                    except Exception:
                        continue

            cargo_holds.append({
                "hold_id": hold_id,
                "exclusive_holds": [x for x in exclusive.split("/") if x],
                "uld_types": [x for x in uld_types.split("/") if x],
                "max_weight": max_weight,
                "arm": arm,
                "cg_coefficient": cg_coef,
            })

        self.cargo_holds = pd.DataFrame(cargo_holds)
        return self.cargo_holds

    def load_flight_params(self) -> Dict[str, float]:
        fp_csv = None
        fp_xlsx = None
        for d in self._candidate_aircraft_dirs():
            p_csv = os.path.join(d, "航班参数.csv")
            p_xlsx = os.path.join(d, "航班参数.xlsx")
            if fp_csv is None and os.path.exists(p_csv):
                fp_csv = p_csv
            if fp_xlsx is None and os.path.exists(p_xlsx):
                fp_xlsx = p_xlsx

        if fp_xlsx:
            df = pd.read_excel(fp_xlsx)
        elif fp_csv:
            try:
                df = pd.read_csv(fp_csv, encoding="gbk")
            except Exception:
                df = pd.read_csv(fp_csv, encoding="utf-8")
        else:
            self.flight_params = {"initial_weight": 40000.0, "initial_cg": 0.0}
            return self.flight_params

        row = df.iloc[0]
        try:
            w1 = float(row.get("dryOperatingWeight", row.get("干操作重量", 40000)))
            c1 = float(row.get("dryOperatingCenter", row.get("干操作重心", 0)))
            w2 = float(row.get("passengerWeight", row.get("旅客重量", 8000)))
            c2 = float(row.get("passengerCenter", row.get("旅客重心", 0)))
        except Exception:
            w1, c1, w2, c2 = 40000.0, 0.0, 8000.0, 0.0

        self.flight_params = {"initial_weight": w1 + w2, "initial_cg": c1 + c2}
        return self.flight_params

    def load_cg_limits(self) -> Dict[str, Any]:
        aft_file = None
        fwd_file = None
        for d in self._candidate_aircraft_dirs():
            p_a = os.path.join(d, "stdZfw_a.csv")
            p_f = os.path.join(d, "stdZfw_f.csv")
            if aft_file is None and os.path.exists(p_a):
                aft_file = p_a
            if fwd_file is None and os.path.exists(p_f):
                fwd_file = p_f

        if aft_file and fwd_file:
            df_aft = pd.read_csv(aft_file, header=None)
            df_fwd = pd.read_csv(fwd_file, header=None)

            weights_aft = df_aft.iloc[:, 1].values.astype(float)
            cg_aft = df_aft.iloc[:, 3].values.astype(float)
            weights_fwd = df_fwd.iloc[:, 1].values.astype(float)
            cg_fwd = df_fwd.iloc[:, 3].values.astype(float)

            aft_interp = interpolate.interp1d(weights_aft, cg_aft, kind="linear", fill_value="extrapolate")
            fwd_interp = interpolate.interp1d(weights_fwd, cg_fwd, kind="linear", fill_value="extrapolate")
        else:
            aft_interp = lambda w: -0.1
            fwd_interp = lambda w: 0.1

        self.cg_limits = {"aft_limit": aft_interp, "fwd_limit": fwd_interp}
        return self.cg_limits

    def load_cargo_data(self) -> pd.DataFrame:
        files = glob.glob(os.path.join(self.cargo_data_dir, "BAKFLGITH_LOADDATA*.csv"))
        if not files:
            raise FileNotFoundError(f"No cargo data found: {self.cargo_data_dir}")

        parts = []
        for f in files:
            for enc in ("utf-8", "gbk", "gb2312", "latin1"):
                try:
                    parts.append(pd.read_csv(f, encoding=enc, header=None))
                    break
                except Exception:
                    continue

        df = pd.concat(parts, ignore_index=True)
        df["flight_number"] = df.iloc[:, 0].astype(str)
        df["fleet"] = df.iloc[:, 1].astype(str).str.upper()
        df["destination"] = df.iloc[:, 2].astype(str).str.upper()
        df["weight"] = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0.0)
        df["content_type"] = df.iloc[:, 4].astype(str).str.upper()
        df["cargo_type"] = df.iloc[:, 7].astype(str) if df.shape[1] > 7 else ""
        df["volume"] = pd.to_numeric(df.iloc[:, 11], errors="coerce").fillna(0.0) if df.shape[1] > 11 else 0.0

        patterns = ["A320", "320", "32A", "32B", "32N"]
        mask = df["fleet"].apply(lambda x: any(p in str(x).upper() for p in patterns))
        df_filtered = df[mask].copy()

        flight_dest_count = df_filtered.groupby("flight_number")["destination"].nunique()
        multi_flights = set(flight_dest_count[flight_dest_count > 1].index)
        df_filtered["is_multi_segment"] = df_filtered["flight_number"].isin(multi_flights)

        self.cargo_data = df_filtered
        return df_filtered

    def get_top_flights(self, n_flights: int = 10, single_only: bool = True) -> List[str]:
        if self.cargo_data is None:
            self.load_cargo_data()
        df = self.cargo_data
        if single_only:
            df = df[~df["is_multi_segment"]]
        flight_counts = df.groupby("flight_number").size().reset_index(name="count")
        flight_counts = flight_counts.sort_values("count", ascending=False)
        return flight_counts["flight_number"].head(n_flights).tolist()

    def get_flight_cargo(self, flight_number: str) -> pd.DataFrame:
        if self.cargo_data is None:
            self.load_cargo_data()
        return self.cargo_data[self.cargo_data["flight_number"] == flight_number].copy()

    def split_cargo(self, cargo_df: pd.DataFrame, max_piece_weight: float = 50, min_piece_weight: float = 10) -> pd.DataFrame:
        split_items = []
        for idx, row in cargo_df.iterrows():
            weight = float(row.get("weight", 0.0))
            volume = float(row.get("volume", 0.0))

            if weight <= max_piece_weight:
                split_items.append({
                    "original_idx": int(idx),
                    "flight_number": row.get("flight_number"),
                    "weight": weight,
                    "volume": volume,
                    "content_type": row.get("content_type", "C"),
                    "cargo_type": row.get("cargo_type", ""),
                    "is_bulk": True,
                    "piece_id": 0,
                    "total_pieces": 1,
                })
            else:
                n_pieces = max(2, int(np.ceil(weight / max_piece_weight)))
                piece_weight = weight / n_pieces
                piece_volume = volume / n_pieces if volume > 0 else 0.0

                if piece_weight < min_piece_weight:
                    n_pieces = max(1, int(weight / min_piece_weight))
                    piece_weight = weight / n_pieces
                    piece_volume = volume / n_pieces if volume > 0 else 0.0

                for p in range(n_pieces):
                    split_items.append({
                        "original_idx": int(idx),
                        "flight_number": row.get("flight_number"),
                        "weight": piece_weight,
                        "volume": piece_volume,
                        "content_type": row.get("content_type", "C"),
                        "cargo_type": row.get("cargo_type", ""),
                        "is_bulk": True,
                        "piece_id": int(p),
                        "total_pieces": int(n_pieces),
                    })
        return pd.DataFrame(split_items)

# --------------------- result helpers ---------------------

def extract_solution_from_result(result: Any) -> Optional[Any]:
    if result is None:
        return None
    if isinstance(result, dict):
        for k in ("solution", "best_solution", "assignment", "alloc", "placements"):
            if k in result:
                return result[k]
        if "evaluation" in result and isinstance(result["evaluation"], dict):
            for k in ("solution", "best_solution"):
                if k in result["evaluation"]:
                    return result["evaluation"][k]
    return None

def summarize_violations(violations: List[Dict[str, Any]], max_items: int = 4) -> str:
    if not violations:
        return "0"
    counts = {}
    for v in violations:
        t = v.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_items]
    return ", ".join([f"{t}:{c}" for t, c in items]) + ("" if len(counts) <= max_items else ", ...")


def create_problem(CargoLoadingProblem, cargo_holds, flight_params, cg_limits, cargo_items,
                   segment_type="single", check_exclusive=True, constraint_level="basic"):
    if CargoLoadingProblem is not None:
        try:
            return CargoLoadingProblem(
                cargo_holds=cargo_holds,
                flight_params=flight_params,
                cg_limits=cg_limits,
                cargo_items=cargo_items,
                segment_type=segment_type,
            )
        except Exception as e:
            print(f"    ⚠️ Using algorithm CargoLoadingProblem failed: {e}. Fallback to LocalCargoLoadingProblem.")

    return LocalCargoLoadingProblem(
        cargo_holds=cargo_holds,
        flight_params=flight_params,
        cg_limits=cg_limits,
        cargo_items=cargo_items,
        segment_type=segment_type,
        check_exclusive=check_exclusive,
        constraint_level=constraint_level,
    )

def run_single_algorithm(algo_class, problem, time_limit: int):
    algo = algo_class(problem, segment_type="single", time_limit=time_limit)
    result = algo.run_with_metrics()

    sol = extract_solution_from_result(result)
    if sol is None and hasattr(algo, "best_solution"):
        sol = getattr(algo, "best_solution")

    violations = None
    try:
        if sol is not None and hasattr(problem, "check_constraints"):
            violations = problem.check_constraints(sol)
    except Exception:
        violations = None

    return result, sol, violations

def print_algo_line(name: str, result: dict, n_items: int, status: str,
                    violations: Optional[List[Dict[str, Any]]] = None):
    if status == "OK" and isinstance(result, dict):
        try:
            gap = float(result["evaluation"]["cg_gap_percent"])
            t = float(result.get("solve_time", np.nan))
            mem = float(result.get("memory_peak_mb", np.nan))
            feasible = bool(result["evaluation"].get("feasible", True))
            vio_cnt = len(violations) if isinstance(violations, list) else None
            vio_sum = summarize_violations(violations) if isinstance(violations, list) else "NA"
            feasible_mark = "✓" if feasible else "✗"
            print(f"      {name:<12} | Gap:{gap:>8.2f}% | Time:{t:>7.2f}s | Mem:{mem:>7.2f}MB | "
                  f"{feasible_mark} | Vio:{vio_cnt if vio_cnt is not None else 'NA':>3} ({vio_sum}) | Items:{n_items}")
        except Exception as e:
            print(f"      {name:<12} | OK but parse error: {e} | Items:{n_items}")
    elif status == "TIMEOUT":
        print(f"      {name:<12} | ⏰ TIMEOUT | Items:{n_items}")
    else:
        print(f"      {name:<12} | ❌ {status} | Items:{n_items}")

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="A320 Experiment 2: Timeout Behavior Characterization (single only)")
    parser.add_argument("--code-root", type=str, default=r"G:\AirCa\code",
                        help="Project code root that contains 'algorithm' folder.")
    parser.add_argument("--aircraft-data-dir", type=str, default=r"G:\AirCa\code\aircraft_data",
                        help="Directory containing aircraft data files (A320.csv, stdZfw_*.csv, 航班参数.*).")
    parser.add_argument("--cargo-data-dir", type=str, default=r"G:\loading_benchmark\bakFlightLoadData\bakFlightLoadData",
                        help="Directory containing BAKFLGITH_LOADDATA*.csv")
    parser.add_argument("--output-dir", type=str, default=r"G:\loading_benchmark\AirCa_output\scaling_analysis",
                        help="Output directory for CSV results.")
    parser.add_argument("--n-flights", type=int, default=4, help="Number of top flights (single-leg) to test.")
    parser.add_argument("--split-threshold", type=int, default=50, help="Split threshold in kg (fixed for this exp).")
    parser.add_argument("--time-limits", type=str, default="5,10,30,60,120",
                        help="Comma-separated algorithm time limits to sweep.")
    parser.add_argument("--extra-timeout-buffer", type=int, default=30,
                        help="Extra seconds added on top of each time_limit to avoid thread cutoff.")
    parser.add_argument("--constraint-level", type=str, choices=["basic", "tight"], default="basic",
                        help="If 'tight', also check CG envelope as a hard constraint (reported in violations).")
    parser.add_argument("--no-exclusive-check", action="store_true",
                        help="Disable exclusive-hold constraint check in violations.")
    parser.add_argument("--algorithms", type=str, default="",
                        help="Optional comma-separated list of algorithm class names to run (default: all found).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("A320 - Experiment 2: Timeout Behavior Characterization")
    print("=" * 90)
    print(f"code_root          : {args.code_root}")
    print(f"aircraft_data_dir  : {args.aircraft_data_dir}")
    print(f"cargo_data_dir     : {args.cargo_data_dir}")
    print(f"output_dir         : {args.output_dir}")
    print(f"n_flights          : {args.n_flights}")
    print(f"split_threshold    : {args.split_threshold}")
    print(f"time_limits        : {args.time_limits}")
    print(f"extra_buffer       : {args.extra_timeout_buffer}s")
    print(f"constraint_level   : {args.constraint_level}")
    print(f"exclusive_check    : {not args.no_exclusive_check}")
    print("=" * 90)

    CargoLoadingProblem, _, algo_classes, import_mode = import_narrow_algorithms(args.code_root)
    print(f"✓ Imported algorithms via: {import_mode}")

    if args.algorithms.strip():
        wanted = {x.strip() for x in args.algorithms.split(",") if x.strip()}
        algo_classes = [c for c in algo_classes if c.__name__ in wanted]
        missing = wanted - set([c.__name__ for c in algo_classes])
        if missing:
            print(f"⚠️ Not found and skipped: {', '.join(sorted(missing))}")

    time_limits = parse_int_list(args.time_limits)

    loader = A320DataLoader(args.aircraft_data_dir, args.cargo_data_dir, aircraft_type="A320")
    cargo_holds = loader.load_cargo_holds()
    flight_params = loader.load_flight_params()
    cg_limits = loader.load_cg_limits()
    loader.load_cargo_data()

    flights = loader.get_top_flights(n_flights=args.n_flights, single_only=True)
    print(f"Loaded holds: {len(cargo_holds)}, flights selected: {len(flights)}")
    if not flights:
        print("No flights found. Exit.")
        return

    results = []
    start = time.time()

    for tl in time_limits:
        print(f"\n--- Time limit = {tl} s ---")
        for fi, flight_num in enumerate(flights, 1):
            original = loader.get_flight_cargo(flight_num)
            if len(original) == 0:
                continue

            split_df = loader.split_cargo(
                original,
                max_piece_weight=args.split_threshold,
                min_piece_weight=max(5, args.split_threshold // 5),
            )
            n_items = len(split_df)
            if n_items == 0:
                continue

            print(f"  Flight [{fi}/{len(flights)}] {flight_num}: {n_items} items")

            problem = create_problem(
                CargoLoadingProblem,
                cargo_holds=cargo_holds,
                flight_params=flight_params,
                cg_limits=cg_limits,
                cargo_items=split_df.reset_index(drop=True),
                segment_type="single",
                check_exclusive=(not args.no_exclusive_check),
                constraint_level=args.constraint_level,
            )

            actual_timeout = tl + int(args.extra_timeout_buffer)

            for algo_class in algo_classes:
                algo_name = algo_class.__name__
                payload, err = run_with_timeout(run_single_algorithm, actual_timeout, algo_class, problem, tl)
                if payload is None:
                    status = "TIMEOUT" if err == "TIMEOUT" else (err or "ERROR")
                    print_algo_line(algo_name, {}, n_items, status, None)
                    results.append({
                        "aircraft_type": "A320",
                        "time_limit": tl,
                        "flight_number": flight_num,
                        "algorithm": algo_name,
                        "n_items": n_items,
                        "cg_gap_percent": None,
                        "actual_time": actual_timeout if status == "TIMEOUT" else None,
                        "memory_mb": None,
                        "feasible": None,
                        "n_violations": None,
                        "violation_summary": None,
                        "status": status,
                    })
                    continue

                result_dict, _, violations = payload
                try:
                    gap = float(result_dict["evaluation"]["cg_gap_percent"])
                    t = float(result_dict.get("solve_time", np.nan))
                    mem = float(result_dict.get("memory_peak_mb", np.nan))
                    feasible = bool(result_dict["evaluation"].get("feasible", True))
                    status = "OK"
                except Exception:
                    gap, t, mem, feasible = None, None, None, None
                    status = "OK_PARSE_FAIL"

                print_algo_line(algo_name, result_dict if isinstance(result_dict, dict) else {}, n_items, "OK", violations)

                results.append({
                    "aircraft_type": "A320",
                    "time_limit": tl,
                    "flight_number": flight_num,
                    "algorithm": algo_name,
                    "n_items": n_items,
                    "cg_gap_percent": gap,
                    "actual_time": t,
                    "memory_mb": mem,
                    "feasible": feasible,
                    "n_violations": len(violations) if isinstance(violations, list) else None,
                    "violation_summary": summarize_violations(violations) if isinstance(violations, list) else None,
                    "status": status,
                })

        pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "a320_timeout_intermediate.csv"), index=False)

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.output_dir, "a320_timeout_behavior_results.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("Experiment 2 finished.")
    print(f"Elapsed: {(time.time() - start)/60:.2f} min")
    print(f"Saved : {out_csv}")
    print("=" * 90)

if __name__ == "__main__":
    main()
