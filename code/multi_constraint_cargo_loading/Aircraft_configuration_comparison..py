#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirCa Unified Runner (Single + Demo only)
- Supported aircraft: A320 / C919 / B777
- Modes:
  1) single : run N single-segment flights under multiple algorithms -> save one combined CSV (+ summary)
  2) demo   : pick ONE flight -> run algorithms -> print cargo -> hold_id mapping
- All hard-coded paths become CLI args, with defaults kept.
- Aircraft static data default moved to: G:\\AirCa\\code\\aircraft_data
- Narrow vs wide algorithms:
  narrow: algorithm.for_narrow.* (fallback algorithm1.*)
  wide  : algorithm.for_wide.*   (fallback algorithm.*)
"""

from __future__ import annotations

import os
import sys
import glob
import time
import math
import argparse
import traceback
import importlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
from scipy import interpolate

pd.options.mode.chained_assignment = None


# =========================
# Defaults (can override by CLI)
# =========================
DEFAULT_CODE_ROOT = r"G:\AirCa\code"
DEFAULT_AIRCRAFT_DATA_DIR = r"G:\AirCa\code\aircraft_data"  # <-- per your request
DEFAULT_CARGO_DATA_DIR = r"G:\loading_benchmark\bakFlightLoadData\bakFlightLoadData"
DEFAULT_OUTPUT_DIR = r"G:\loading_benchmark\AirCa_output"

AIRCRAFT_CONFIGS = {

    "B777": {
        "type": "wide",
        "folder": "B777",
    },
    "A320": {
        "type": "narrow",
        "folder": "A320",
    },
    "C919": {
        "type": "narrow",
        "folder": "C919",
    },

}


# =========================
# Helpers
# =========================
def ensure_sys_path(code_root: str) -> None:
    """Ensure code_root is on sys.path so `algorithm` package can be imported."""
    code_root = os.path.abspath(code_root)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def read_csv_multi_encoding(path: str, header=None) -> pd.DataFrame:
    """Try multiple encodings for messy CSVs."""
    last_err = None
    for enc in ["utf-8", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, header=header, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    # last try
    try:
        return pd.read_csv(path, header=header)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}. Last err: {last_err}, final err: {e}")


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pick_base_path(aircraft_data_dir: str, aircraft: str) -> str:
    folder = AIRCRAFT_CONFIGS[aircraft]["folder"]
    return os.path.join(aircraft_data_dir, folder)


def find_aircraft_csv(base_path: str, aircraft: str) -> str:
    cand = os.path.join(base_path, f"{aircraft}.csv")
    if os.path.exists(cand):
        return cand
    # fallback: first csv not containing zfw
    for f in os.listdir(base_path):
        if f.lower().endswith(".csv") and "zfw" not in f.lower():
            return os.path.join(base_path, f)
    raise FileNotFoundError(f"Cannot find aircraft config csv for {aircraft} under: {base_path}")


def import_symbols(module_name: str, symbols: List[str]) -> Dict[str, Any]:
    mod = importlib.import_module(module_name)
    out = {}
    for s in symbols:
        if hasattr(mod, s):
            out[s] = getattr(mod, s)
    return out


def load_algorithm_classes(aircraft: str) -> List[type]:
    """
    Return ordered algorithm classes:
      exact: MILP, MINLP, QP, DP, CP
      heuristic: GA, PSO, CS, ACO, ABC, MBO
    """
    a_type = AIRCRAFT_CONFIGS[aircraft]["type"]
    ordered = ["MILP", "MINLP", "QP", "DP", "CP", "GA", "PSO", "CS", "ACO", "ABC", "MBO"]

    if a_type == "narrow":
        candidates = [
            ("algorithm.for_narrow.milp_algorithm1", ["MILP"]),
            ("algorithm.for_narrow.exact_algorithms1", ["MINLP", "QP", "DP", "CP"]),
            ("algorithm.for_narrow.heuristic_algorithms1", ["GA", "PSO", "CS", "ACO", "ABC", "MBO"]),
        ]
        fallback = [
            ("algorithm1.milp_algorithm1", ["MILP"]),
            ("algorithm1.exact_algorithms1", ["MINLP", "QP", "DP", "CP"]),
            ("algorithm1.heuristic_algorithms1", ["GA", "PSO", "CS", "ACO", "ABC", "MBO"]),
        ]
    else:
        candidates = [
            ("algorithm.for_wide.exact_algorithms", ["MILP", "MINLP", "QP", "DP", "CP"]),
            ("algorithm.for_wide.heuristic_algorithms", ["GA", "PSO", "CS", "ACO", "ABC", "MBO"]),
        ]
        fallback = [
            ("algorithm.exact_algorithms", ["MILP", "MINLP", "QP", "DP", "CP"]),
            ("algorithm.heuristic_algorithms", ["GA", "PSO", "CS", "ACO", "ABC", "MBO"]),
        ]

    def try_load(mod_specs) -> Dict[str, type]:
        pool = {}
        for mod, syms in mod_specs:
            try:
                d = import_symbols(mod, syms)
                pool.update(d)
            except Exception:
                continue
        return pool

    pool = try_load(candidates)
    if not pool:
        pool = try_load(fallback)

    if not pool:
        raise ModuleNotFoundError(
            f"Cannot import algorithms for {aircraft}. "
            f"Please check your package layout under algorithm/for_narrow and algorithm/for_wide."
        )

    algos = []
    for name in ordered:
        if name in pool:
            algos.append(pool[name])
    # if some missing, append what exists
    for k, v in pool.items():
        if v not in algos:
            algos.append(v)
    return algos


def instantiate_algorithm(algo_cls: type, problem: Any, segment_type: str = "single", time_limit: Optional[int] = None) -> Any:
    """
    Try multiple constructor signatures, because your algo classes may differ slightly.
    """
    attempts = []
    if time_limit is not None:
        attempts.extend([
            lambda: algo_cls(problem, segment_type=segment_type, time_limit=time_limit),
            lambda: algo_cls(problem, time_limit=time_limit),
        ])
    attempts.extend([
        lambda: algo_cls(problem, segment_type=segment_type),
        lambda: algo_cls(problem),
    ])

    last_err = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore


def run_algo(algo_obj: Any) -> Dict[str, Any]:
    """
    Prefer run_with_metrics(); else run().
    """
    if hasattr(algo_obj, "run_with_metrics"):
        out = algo_obj.run_with_metrics()
        return out if isinstance(out, dict) else {"output": out}
    if hasattr(algo_obj, "run"):
        out = algo_obj.run()
        return out if isinstance(out, dict) else {"solution": out}
    raise AttributeError("Algorithm has neither run_with_metrics() nor run().")


def extract_solution(result: Dict[str, Any], algo_obj: Any = None) -> Any:
    """
    Try to find solution in result dict or algo object.
    """
    keys = ["solution", "best_solution", "assignment", "best_assignment", "loading_plan", "plan"]
    for k in keys:
        if k in result:
            return result[k]
    if "output" in result and isinstance(result["output"], dict):
        return extract_solution(result["output"], algo_obj)

    if algo_obj is not None:
        for attr in ["solution", "best_solution", "assignment", "best_assignment"]:
            if hasattr(algo_obj, attr):
                return getattr(algo_obj, attr)
    return None


def normalize_solution(sol: Any, n_items: int) -> Dict[int, int]:
    """
    Convert solution into {item_idx: hold_idx}, hold_idx=-1 means not loaded.
    """
    if sol is None:
        return {i: -1 for i in range(n_items)}

    if isinstance(sol, dict):
        out = {}
        for k, v in sol.items():
            try:
                out[int(k)] = int(v)
            except Exception:
                continue
        for i in range(n_items):
            out.setdefault(i, -1)
        return out

    if isinstance(sol, (list, tuple, np.ndarray, pd.Series)):
        arr = list(sol)
        out = {}
        for i in range(n_items):
            out[i] = int(arr[i]) if i < len(arr) and arr[i] is not None else -1
        return out

    # unknown type
    return {i: -1 for i in range(n_items)}


# =========================
# Narrow-body loader + problem (A320 / C919)
# =========================
class NarrowDataLoader:
    """
    Narrow-body data loader (A320 / C919).
    Kept consistent with your original a320_data_loader.py / c919_data_loader.py behavior.
    """

    def __init__(self, aircraft: str, base_path: str, cargo_data_path: str):
        if aircraft not in ("A320", "C919"):
            raise ValueError("NarrowDataLoader supports only A320 / C919.")
        self.aircraft = aircraft
        self.base_path = base_path
        self.cargo_data_path = cargo_data_path

        self.cargo_holds = None
        self.flight_params = None
        self.cg_limits = None
        self.cargo_data = None

    def load_cargo_holds(self) -> pd.DataFrame:
        filepath = find_aircraft_csv(self.base_path, self.aircraft)
        df = read_csv_multi_encoding(filepath, header=None)

        cargo_holds = []
        for _, row in df.iterrows():
            hold_id = str(row[1]).strip() if pd.notna(row[1]) else None
            if hold_id is None or hold_id == "nan":
                continue

            exclusive = str(row[2]).strip() if len(row) > 2 and pd.notna(row[2]) else ""
            uld_types = str(row[3]).strip() if len(row) > 3 and pd.notna(row[3]) else ""
            max_weight = safe_float(row[4], 0.0) if len(row) > 4 else 0.0

            # cg coef: last numeric
            cg_coef = None
            for i in range(len(row) - 1, -1, -1):
                if pd.notna(row[i]) and row[i] != "":
                    try:
                        cg_coef = float(row[i])
                        break
                    except Exception:
                        continue

            # arm: try columns 8/9/10
            arm = None
            for i in [8, 9, 10]:
                if i < len(row) and pd.notna(row[i]):
                    try:
                        arm = float(row[i])
                        break
                    except Exception:
                        continue

            cargo_holds.append({
                "hold_id": hold_id,
                "exclusive_holds": exclusive.split("/") if exclusive and exclusive != "////" else [],
                "uld_types": uld_types.split("/") if uld_types and uld_types != "////" else [],
                "max_weight": max_weight,
                "arm": arm,
                "cg_coefficient": cg_coef,
            })

        self.cargo_holds = pd.DataFrame(cargo_holds)
        return self.cargo_holds

    def load_flight_params(self) -> Dict[str, float]:
        filepath = os.path.join(self.base_path, "航班参数.csv")
        df = read_csv_multi_encoding(filepath, header=0)

        row = df.iloc[0]
        # align with your original naming
        weight1 = float(row["dryOperatingWeight"])
        cg1 = float(row["dryOperatingCenter"])
        weight2 = float(row["passengerWeight"])
        cg2 = float(row["passengerCenter"])

        initial_weight = weight1 + weight2
        initial_cg = cg1 + cg2

        self.flight_params = {
            "initial_weight": initial_weight,
            "initial_cg": initial_cg,
            "dry_operating_weight": weight1,
            "dry_operating_cg": cg1,
            "passenger_weight": weight2,
            "passenger_cg": cg2,
            "zfw": float(row.get("zfw", 0.0)),
            "zfw_cg": float(row.get("lizfw", 0.0)),
        }
        return self.flight_params

    def load_cg_limits(self) -> Dict[str, Any]:
        aft_filepath = os.path.join(self.base_path, "stdZfw_a.csv")
        fwd_filepath = os.path.join(self.base_path, "stdZfw_f.csv")

        df_aft = read_csv_multi_encoding(aft_filepath, header=None)
        weights_aft = df_aft.iloc[:, 1].values.astype(float)
        cg_aft = df_aft.iloc[:, 3].values.astype(float)

        df_fwd = read_csv_multi_encoding(fwd_filepath, header=None)
        weights_fwd = df_fwd.iloc[:, 1].values.astype(float)
        cg_fwd = df_fwd.iloc[:, 3].values.astype(float)

        aft_interp = interpolate.interp1d(weights_aft, cg_aft, kind="linear", fill_value="extrapolate")
        fwd_interp = interpolate.interp1d(weights_fwd, cg_fwd, kind="linear", fill_value="extrapolate")

        self.cg_limits = {
            "aft_limit": aft_interp,
            "fwd_limit": fwd_interp,
            "weights_aft": weights_aft,
            "cg_aft": cg_aft,
            "weights_fwd": weights_fwd,
            "cg_fwd": cg_fwd,
        }
        return self.cg_limits

    def load_cargo_data(self) -> pd.DataFrame:
        cargo_files = glob.glob(os.path.join(self.cargo_data_path, "BAKFLGITH_LOADDATA*.csv"))
        if not cargo_files:
            raise FileNotFoundError(f"未找到货物数据文件: {self.cargo_data_path}")

        all_data = []
        for cargo_file in cargo_files:
            df = read_csv_multi_encoding(cargo_file, header=None)
            all_data.append(df)

        df = pd.concat(all_data, ignore_index=True)

        # mimic your original filtering logic:
        df["aircraft_type"] = df.iloc[:, 1].astype(str).apply(
            lambda x: x.split("-")[0].upper() if "-" in str(x) else str(x).upper()
        )

        if self.aircraft == "A320":
            def normalize_type(t):
                t = str(t).upper().strip()
                if t in ["320", "A320", "A32", "32A", "32B", "32N"]:
                    return "A320"
                if t in ["321", "A321", "A21"]:
                    return "A320"
                if t in ["319", "A319"]:
                    return "A320"
                return t
        else:  # C919
            def normalize_type(t):
                t = str(t).upper().strip()
                if t in ["919", "C919", "A32", "32A", "32B", "32N"]:
                    return "C919"
                if t in ["321", "A321", "A21"]:
                    return "C919"
                if t in ["319", "A319"]:
                    return "C919"
                return t

        df["aircraft_type"] = df["aircraft_type"].apply(normalize_type)
        df_filtered = df[df["aircraft_type"] == self.aircraft].copy()

        # parse key fields
        df_filtered["flight_number"] = df_filtered.iloc[:, 0].astype(str)
        df_filtered["fleet"] = df_filtered.iloc[:, 1].astype(str)
        df_filtered["destination"] = df_filtered.iloc[:, 2].astype(str).str.upper()
        df_filtered["weight"] = pd.to_numeric(df_filtered.iloc[:, 3], errors="coerce").fillna(0)
        df_filtered["content_type"] = df_filtered.iloc[:, 4].astype(str).str.upper()  # C/B/M

        # col 8 cargo type / uld marker
        if df_filtered.shape[1] > 7:
            df_filtered["cargo_type"] = df_filtered.iloc[:, 7].astype(str)
            df_filtered["is_bulk"] = df_filtered["cargo_type"].apply(lambda x: x == "" or x == "nan" or pd.isna(x))
        else:
            df_filtered["is_bulk"] = True

        # col 12 volume
        if df_filtered.shape[1] > 11:
            df_filtered["volume"] = pd.to_numeric(df_filtered.iloc[:, 11], errors="coerce").fillna(0)
        else:
            df_filtered["volume"] = 0

        # single vs multi: destination count
        flight_dest_count = df_filtered.groupby("flight_number")["destination"].nunique()
        multi_segment_flights = set(flight_dest_count[flight_dest_count > 1].index)
        df_filtered["is_multi_segment"] = df_filtered["flight_number"].isin(multi_segment_flights)

        self.cargo_data = df_filtered
        print(f"[{self.aircraft}] cargo records loaded: {len(df_filtered)}")
        return self.cargo_data

    def split_cargo_by_segment(self, n_single=100, n_multi=0) -> Dict[str, List[str]]:
        if self.cargo_data is None:
            raise ValueError("请先 load_cargo_data()")

        flight_cargo_count = self.cargo_data.groupby("flight_number").size().reset_index(name="cargo_count")

        single_flights_all = self.cargo_data[~self.cargo_data["is_multi_segment"]]["flight_number"].unique()
        single_counts = flight_cargo_count[flight_cargo_count["flight_number"].isin(single_flights_all)]
        single_counts = single_counts.sort_values("cargo_count", ascending=False)
        single_flights = single_counts["flight_number"].head(n_single).tolist()

        # we do not need multi in this merged version, but keep a placeholder
        return {"single_flights": single_flights, "multi_flights": []}

    def get_flight_cargo(self, flight_number: str) -> pd.DataFrame:
        if self.cargo_data is None:
            raise ValueError("请先 load_cargo_data()")
        return self.cargo_data[self.cargo_data["flight_number"] == flight_number].copy().reset_index(drop=True)


class NarrowCargoLoadingProblem:
    """Same interface as your original CargoLoadingProblem in a320/c919 loader."""

    def __init__(self, cargo_holds, flight_params, cg_limits, cargo_items, segment_type="single"):
        self.cargo_holds = cargo_holds
        self.flight_params = flight_params
        self.cg_limits = cg_limits
        self.cargo_items = cargo_items
        self.segment_type = segment_type

        self.holds = self._get_unique_holds()
        self.n_holds = len(self.holds)
        self.n_items = len(cargo_items)

        self.initial_weight = flight_params["initial_weight"]
        self.initial_cg = flight_params["initial_cg"]

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
        aft = self.cg_limits["aft_limit"](total_weight)
        fwd = self.cg_limits["fwd_limit"](total_weight)
        optimal = aft + (fwd - aft) * (1 / 3)
        return float(optimal), float(aft), float(fwd)

    def calculate_cg(self, solution):
        total_moment = self.initial_cg
        total_weight = self.initial_weight

        if isinstance(solution, dict):
            items = solution.items()
        else:
            items = enumerate(solution)

        for item_idx, hold_idx in items:
            if hold_idx >= 0 and hold_idx < len(self.holds):
                item = self.cargo_items.iloc[item_idx]
                hold = self.holds[hold_idx]
                weight = item["weight"]
                cg_coef = hold["cg_coefficient"]
                total_weight += weight
                total_moment += weight * cg_coef

        return total_moment, total_weight

    def evaluate_solution(self, solution):
        cg, total_weight = self.calculate_cg(solution)
        optimal_cg, aft_limit, fwd_limit = self.get_optimal_cg(total_weight)

        cg_gap = abs(cg - optimal_cg)
        cg_gap_percent = cg_gap / abs(optimal_cg) * 100 if optimal_cg != 0 else 0

        feasible = (aft_limit <= cg <= fwd_limit) and (len(self.check_constraints(solution)) == 0)

        revenue = 0.0
        if isinstance(solution, dict):
            items = solution.items()
        else:
            items = enumerate(solution)
        for item_idx, hold_idx in items:
            if hold_idx >= 0:
                revenue += float(self.cargo_items.iloc[item_idx]["weight"])

        return {
            "cg": float(cg),
            "total_weight": float(total_weight),
            "optimal_cg": float(optimal_cg),
            "cg_gap": float(cg_gap),
            "cg_gap_percent": float(cg_gap_percent),
            "aft_limit": float(aft_limit),
            "fwd_limit": float(fwd_limit),
            "feasible": bool(feasible),
            "revenue": float(revenue),
        }

    def check_constraints(self, solution):
        violations = []
        hold_weights = {i: 0.0 for i in range(self.n_holds)}

        if isinstance(solution, dict):
            items = solution.items()
            used_holds = set([h for h in solution.values() if isinstance(h, (int, np.integer)) and h >= 0])
        else:
            items = enumerate(solution)
            used_holds = set([h for h in solution if isinstance(h, (int, np.integer)) and h >= 0])

        for item_idx, hold_idx in items:
            if hold_idx >= 0 and hold_idx < self.n_holds:
                hold_weights[hold_idx] += float(self.cargo_items.iloc[item_idx]["weight"])

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


        # weight constraint
        for hold_idx, w in hold_weights.items():
            if w > float(self.holds[hold_idx]["max_weight"]):
                violations.append({
                    "type": "weight",
                    "hold": self.holds[hold_idx]["hold_id"],
                    "actual": float(w),
                    "limit": float(self.holds[hold_idx]["max_weight"]),
                })

        # exclusive holds
        for hold_idx in used_holds:
            hold = self.holds[hold_idx]
            exclusive = hold.get("exclusive_holds", [])
            for exc_hold_id in exclusive:
                if exc_hold_id:
                    for other_idx, other_hold in enumerate(self.holds):
                        if other_hold["hold_id"] == exc_hold_id and other_idx in used_holds:
                            violations.append({
                                "type": "exclusive",
                                "hold1": hold["hold_id"],
                                "hold2": exc_hold_id,
                            })

        return violations


# =========================
# Wide-body loader + problem (B777)
# (Copied from your b777_data_loader.py final version, with minimal edits)
# =========================
class B777DataLoader:
    """B777数据加载器"""

    def __init__(self, base_path=None, cargo_data_path=None):
        self.base_path = base_path or r"G:\loading_benchmark\B777"
        self.cargo_data_path = cargo_data_path or r"G:\loading_benchmark\bakFlightLoadData\bakFlightLoadData"

        self.aircraft_type = "B777"
        self.fleet_pattern = ["B777", "777", "77A", "77B", "77W"]
        self.is_widebody = True

        self.cargo_holds = None
        self.flight_params = None
        self.cg_limits = None
        self.cargo_data = None

    def load_cargo_holds(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.base_path, "B777.csv")

        if not os.path.exists(filepath):
            for f in os.listdir(self.base_path):
                if f.endswith(".csv") and "zfw" not in f.lower():
                    filepath = os.path.join(self.base_path, f)
                    break

        df = read_csv_multi_encoding(filepath, header=None)

        cargo_holds = []
        for _, row in df.iterrows():
            hold_id = str(row[1]).strip() if pd.notna(row[1]) else None
            if hold_id is None or hold_id == "nan":
                continue

            exclusive = str(row[2]).strip() if pd.notna(row[2]) else ""
            uld_types = str(row[3]).strip() if pd.notna(row[3]) else ""
            max_weight = float(row[4]) if pd.notna(row[4]) else 0

            cg_coef = None
            for i in range(len(row) - 1, -1, -1):
                if pd.notna(row[i]) and row[i] != "":
                    try:
                        cg_coef = float(row[i])
                        break
                    except Exception:
                        continue

            arm = None
            for i in [8, 9, 10]:
                if i < len(row) and pd.notna(row[i]):
                    try:
                        arm = float(row[i])
                        break
                    except Exception:
                        continue

            uld_type_list = [x.strip() for x in uld_types.split("/") if x.strip() and x.strip() != "////"]
            exclusive_list = [x.strip() for x in exclusive.split("/") if x.strip() and x.strip() != "////"]

            cargo_holds.append({
                "hold_id": hold_id,
                "exclusive_holds": exclusive_list,
                "uld_types": uld_type_list,
                "max_weight": max_weight,
                "arm": arm,
                "cg_coefficient": cg_coef,
            })

        self.cargo_holds = pd.DataFrame(cargo_holds)
        return self.cargo_holds

    def load_flight_params(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.base_path, "航班参数.csv")

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding="gbk")
            except Exception:
                df = read_csv_multi_encoding(filepath, header=0)
        else:
            filepath_xlsx = os.path.join(self.base_path, "航班参数.xlsx")
            if os.path.exists(filepath_xlsx):
                df = pd.read_excel(filepath_xlsx)
            else:
                self.flight_params = {"initial_weight": 150000, "initial_cg": 0}
                return self.flight_params

        row = df.iloc[0]
        try:
            weight1 = float(row.get("dryOperatingWeight", row.get("干操作重量", 150000)))
            cg1 = float(row.get("dryOperatingCenter", row.get("干操作重心", 0)))
            weight2 = float(row.get("passengerWeight", row.get("旅客重量", 20000)))
            cg2 = float(row.get("passengerCenter", row.get("旅客重心", 0)))
        except Exception:
            weight1, cg1, weight2, cg2 = 150000, 0, 20000, 0

        self.flight_params = {"initial_weight": weight1 + weight2, "initial_cg": cg1 + cg2}
        return self.flight_params

    def load_cg_limits(self):
        aft_file = os.path.join(self.base_path, "stdZfw_a.csv")
        fwd_file = os.path.join(self.base_path, "stdZfw_f.csv")

        if os.path.exists(aft_file) and os.path.exists(fwd_file):
            df_aft = read_csv_multi_encoding(aft_file, header=None)
            df_fwd = read_csv_multi_encoding(fwd_file, header=None)

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

    def load_cargo_data(self):
        cargo_files = glob.glob(os.path.join(self.cargo_data_path, "BAKFLGITH_LOADDATA*.csv"))
        if not cargo_files:
            raise FileNotFoundError(f"未找到货物数据: {self.cargo_data_path}")

        all_data = []
        for cargo_file in cargo_files:
            df = read_csv_multi_encoding(cargo_file, header=None)
            all_data.append(df)

        df = pd.concat(all_data, ignore_index=True)

        df["flight_number"] = df.iloc[:, 0].astype(str)
        df["fleet"] = df.iloc[:, 1].astype(str).str.upper()
        df["destination"] = df.iloc[:, 2].astype(str).str.upper()
        df["weight"] = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0)
        df["content_type"] = df.iloc[:, 4].astype(str).str.upper()

        if df.shape[1] > 7:
            df["uld_type"] = df.iloc[:, 7].astype(str).str.strip()
            df["uld_type"] = df["uld_type"].apply(lambda x: "" if x == "nan" or pd.isna(x) else x)
        else:
            df["uld_type"] = ""

        if df.shape[1] > 11:
            df["volume"] = pd.to_numeric(df.iloc[:, 11], errors="coerce").fillna(0)
        else:
            df["volume"] = 0

        df["is_bulk"] = df["uld_type"].apply(lambda x: x == "" or x == "nan" or pd.isna(x))

        mask = df["fleet"].apply(lambda x: any(p in str(x).upper() for p in self.fleet_pattern))
        df_filtered = df[mask].copy()

        flight_dest_count = df_filtered.groupby("flight_number")["destination"].nunique()
        multi_flights = set(flight_dest_count[flight_dest_count > 1].index)
        df_filtered["is_multi_segment"] = df_filtered["flight_number"].isin(multi_flights)

        flight_weights = df_filtered.groupby("flight_number")["weight"].sum().reset_index()
        flight_weights.columns = ["flight_number", "total_weight"]
        df_filtered = df_filtered.merge(flight_weights, on="flight_number")

        self.cargo_data = df_filtered
        print(f"[B777] cargo records loaded: {len(df_filtered)}")
        return self.cargo_data

    def split_cargo_by_segment(self, n_single=100, n_multi=0):
        df = self.cargo_data[~self.cargo_data["is_multi_segment"]]
        flights = df.groupby("flight_number")["weight"].sum().reset_index()
        flights = flights.sort_values("weight", ascending=False)
        return {"single_flights": flights["flight_number"].head(n_single).tolist(), "multi_flights": []}

    def get_flight_cargo(self, flight_number):
        return self.cargo_data[self.cargo_data["flight_number"] == flight_number].copy().reset_index(drop=True)


class WideCargoLoadingProblem:
    """
    Wide-body cargo loading problem (B777)
    This is essentially the final CargoLoadingProblem in your b777_data_loader.py.
    """

    def __init__(self, cargo_holds, flight_params, cg_limits, cargo_items,
                 segment_type="single", cg_weight=0.5, revenue_weight=0.5, is_widebody=True):
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

        self.initial_weight = flight_params["initial_weight"]
        self.initial_cg = flight_params["initial_cg"]

        self._max_possible_revenue = self._estimate_max_revenue()

    def _estimate_max_revenue(self):
        total = 0.0
        for idx in range(len(self.cargo_items)):
            w = float(self.cargo_items.iloc[idx]["weight"])
            rate = 5.44 if w >= 1000 else (6.27 if w >= 500 else 9.07)
            total += max(70.0, w * rate)
        return max(total, 1.0)

    def _get_unique_holds(self):
        holds = []
        seen = set()
        for _, row in self.cargo_holds.iterrows():
            hold_id = row["hold_id"]
            if hold_id not in seen:
                holds.append(row.to_dict())
                seen.add(hold_id)
        return holds

    def is_hold_compatible(self, item, hold):
        item_uld_type = ""
        if isinstance(item, dict):
            item_uld_type = item.get("uld_type", "")
        else:
            try:
                item_uld_type = str(item.get("uld_type", ""))
            except Exception:
                item_uld_type = ""

        item_uld_type = str(item_uld_type).strip() if item_uld_type else ""
        allowed_types = hold.get("uld_types", [])

        if not allowed_types or not item_uld_type:
            return True
        return item_uld_type in allowed_types

    def get_optimal_cg(self, total_weight):
        aft = float(self.cg_limits["aft_limit"](total_weight))
        fwd = float(self.cg_limits["fwd_limit"](total_weight))
        optimal = aft + (fwd - aft) * (1 / 3)
        return float(optimal), float(aft), float(fwd)

    def calculate_cg(self, solution):
        total_moment = self.initial_cg
        total_weight = self.initial_weight

        if isinstance(solution, dict):
            items = solution.items()
        else:
            items = enumerate(solution)

        for item_idx, hold_idx in items:
            if hold_idx >= 0 and hold_idx < len(self.holds):
                item = self.cargo_items.iloc[item_idx]
                hold = self.holds[hold_idx]
                w = float(item["weight"])
                cg_coef = hold["cg_coefficient"]
                if cg_coef is not None:
                    total_weight += w
                    total_moment += w * float(cg_coef)

        return float(total_moment), float(total_weight)

    def calculate_cargo_revenue(self, weight):
        MIN_CHARGE = 70.0
        RATE_TABLE = [
            (44, 12.66),
            (99, 9.74),
            (299, 9.07),
            (499, 7.16),
            (999, 6.27),
            (float("inf"), 5.44),
        ]
        rate = RATE_TABLE[-1][1]
        for upper, r in RATE_TABLE:
            if weight <= upper:
                rate = r
                break
        return max(MIN_CHARGE, weight * rate)

    def calculate_profit(self, gross_revenue, cg_gap_percent):
        penalty_factor = 0.5 * (1 - math.exp(-cg_gap_percent / 50.0))
        return gross_revenue * (1 - penalty_factor)

    def evaluate_solution(self, solution):
        cg, total_weight = self.calculate_cg(solution)
        optimal_cg, aft_limit, fwd_limit = self.get_optimal_cg(total_weight)

        cg_gap = abs(cg - optimal_cg)
        cg_gap_percent = cg_gap / abs(optimal_cg) * 100 if optimal_cg != 0 else 0

        violations = self.check_constraints(solution)
        cg_feasible = aft_limit <= cg <= fwd_limit
        feasible = cg_feasible and len(violations) == 0

        if isinstance(solution, dict):
            items = solution.items()
        else:
            items = enumerate(solution)

        gross_revenue = 0.0
        for item_idx, hold_idx in items:
            if hold_idx >= 0:
                w = float(self.cargo_items.iloc[item_idx]["weight"])
                gross_revenue += self.calculate_cargo_revenue(w)

        profit = self.calculate_profit(gross_revenue, cg_gap_percent)

        return {
            "cg": float(cg),
            "total_weight": float(total_weight),
            "optimal_cg": float(optimal_cg),
            "cg_gap": float(cg_gap),
            "cg_gap_percent": float(cg_gap_percent),
            "aft_limit": float(aft_limit),
            "fwd_limit": float(fwd_limit),
            "feasible": bool(feasible),
            "gross_revenue": float(gross_revenue),
            "revenue": float(profit),
        }

    def check_constraints(self, solution):
        violations = []

        hold_weights = {i: 0.0 for i in range(self.n_holds)}
        hold_items = {i: [] for i in range(self.n_holds)}

        if isinstance(solution, dict):
            items = solution.items()
        else:
            items = enumerate(solution)

        for item_idx, hold_idx in items:
            if hold_idx >= 0 and hold_idx < self.n_holds:
                item = self.cargo_items.iloc[item_idx]
                hold_weights[hold_idx] += float(item["weight"])
                hold_items[hold_idx].append(item_idx)

        # ULD compatibility
        for hold_idx, item_indices in hold_items.items():
            if not item_indices:
                continue
            hold = self.holds[hold_idx]
            allowed_types = hold.get("uld_types", [])
            if not allowed_types:
                continue
            for item_idx in item_indices:
                item = self.cargo_items.iloc[item_idx]
                item_uld = str(item.get("uld_type", "")).strip() if "uld_type" in item else ""
                if item_uld and item_uld not in allowed_types:
                    violations.append({
                        "type": "uld_incompatible",
                        "hold": hold["hold_id"],
                        "hold_idx": hold_idx,
                        "item_idx": item_idx,
                        "allowed_types": allowed_types,
                        "item_uld_type": item_uld,
                    })

        # weight constraint
        for hold_idx, w in hold_weights.items():
            if w > float(self.holds[hold_idx]["max_weight"]):
                violations.append({
                    "type": "weight",
                    "hold": self.holds[hold_idx]["hold_id"],
                    "hold_idx": hold_idx,
                    "actual": float(w),
                    "limit": float(self.holds[hold_idx]["max_weight"]),
                })

        # exclusive holds
        checked_pairs = set()
        for hold_idx, hold in enumerate(self.holds):
            if hold_items[hold_idx] and hold.get("exclusive_holds"):
                for excl_hold_id in hold["exclusive_holds"]:
                    if excl_hold_id:
                        for other_idx, other_hold in enumerate(self.holds):
                            if other_hold["hold_id"] == excl_hold_id and hold_items[other_idx]:
                                pair = tuple(sorted([hold_idx, other_idx]))
                                if pair not in checked_pairs:
                                    violations.append({
                                        "type": "exclusive",
                                        "hold1": hold["hold_id"],
                                        "hold1_idx": hold_idx,
                                        "hold2": excl_hold_id,
                                        "hold2_idx": other_idx,
                                    })
                                    checked_pairs.add(pair)

        return violations


# =========================
# Runner
# =========================
def build_loader_and_problem(aircraft: str, base_path: str, cargo_data_dir: str):
    if aircraft in ("A320", "C919"):
        loader = NarrowDataLoader(aircraft, base_path, cargo_data_dir)
        loader.load_cargo_holds()
        loader.load_flight_params()
        loader.load_cg_limits()
        loader.load_cargo_data()
        return loader, NarrowCargoLoadingProblem
    elif aircraft == "B777":
        loader = B777DataLoader(base_path=base_path, cargo_data_path=cargo_data_dir)
        loader.load_cargo_holds()
        loader.load_flight_params()
        loader.load_cg_limits()
        loader.load_cargo_data()
        return loader, WideCargoLoadingProblem
    else:
        raise ValueError(f"Unsupported aircraft: {aircraft}")


def run_single(
    aircraft: str,
    loader: Any,
    problem_cls: Any,
    algos: List[type],
    n_flights: int,
    time_limit: Optional[int],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    split = loader.split_cargo_by_segment(n_single=n_flights, n_multi=0)
    flights = split["single_flights"]
    print(f"\n[{aircraft}] Selected single flights: {len(flights)}")

    rows = []
    for i, flight_no in enumerate(flights, start=1):
        print(f"\n[{aircraft}] Flight {i}/{len(flights)}: {flight_no}")
        flight_cargo = loader.get_flight_cargo(flight_no)
        if len(flight_cargo) == 0:
            print("  skip: no cargo")
            continue

        problem = problem_cls(
            cargo_holds=loader.cargo_holds,
            flight_params=loader.flight_params,
            cg_limits=loader.cg_limits,
            cargo_items=flight_cargo.reset_index(drop=True),
            segment_type="single",
        )

        print(f"  items={problem.n_items}, holds={problem.n_holds}")

        for algo_cls in algos:
            algo_name = getattr(algo_cls, "__name__", str(algo_cls))
            try:
                algo = instantiate_algorithm(algo_cls, problem, segment_type="single", time_limit=time_limit)
                result = run_algo(algo)

                eval_dict = result.get("evaluation", None)
                if eval_dict is None and hasattr(problem, "evaluate_solution"):
                    sol = extract_solution(result, algo)
                    sol_map = normalize_solution(sol, problem.n_items)
                    eval_dict = problem.evaluate_solution(sol_map)

                solve_time = result.get("solve_time", None)
                mem_mb = result.get("memory_peak_mb", None)

                cg_gap_percent = None
                feasible = None
                revenue = None
                total_weight = None

                if isinstance(eval_dict, dict):
                    cg_gap_percent = eval_dict.get("cg_gap_percent", None)
                    feasible = eval_dict.get("feasible", None)
                    revenue = eval_dict.get("revenue", None)
                    total_weight = eval_dict.get("total_weight", None)

                print(
                    f"  {getattr(algo, 'name', algo_name)}: "
                    f"Gap={cg_gap_percent if cg_gap_percent is not None else 'NA'}%, "
                    f"Time={solve_time if solve_time is not None else 'NA'}s, "
                    f"Mem={mem_mb if mem_mb is not None else 'NA'}MB"
                )

                rows.append({
                    "aircraft": aircraft,
                    "flight_number": flight_no,
                    "algo": getattr(algo, "name", algo_name),
                    "n_items": int(problem.n_items),
                    "n_holds": int(problem.n_holds),
                    "cg_gap_percent": cg_gap_percent,
                    "feasible": feasible,
                    "revenue_or_profit": revenue,
                    "total_weight": total_weight,
                    "solve_time_s": solve_time,
                    "memory_peak_mb": mem_mb,
                })

            except Exception as e:
                print(f"  {algo_name} Error: {e}")
                rows.append({
                    "aircraft": aircraft,
                    "flight_number": flight_no,
                    "algo": algo_name,
                    "n_items": int(problem.n_items),
                    "n_holds": int(problem.n_holds),
                    "cg_gap_percent": None,
                    "feasible": None,
                    "revenue_or_profit": None,
                    "total_weight": None,
                    "solve_time_s": None,
                    "memory_peak_mb": None,
                    "error": str(e),
                })

    df = pd.DataFrame(rows)
    tag = timestamp_str()
    out_csv = os.path.join(out_dir, f"unified_single_{aircraft}_{tag}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    # summary
    if len(df) > 0:
        agg = df.groupby(["aircraft", "algo"], dropna=False).agg(
            avg_gap=("cg_gap_percent", "mean"),
            std_gap=("cg_gap_percent", "std"),
            avg_time=("solve_time_s", "mean"),
            std_time=("solve_time_s", "std"),
            avg_mem=("memory_peak_mb", "mean"),
            std_mem=("memory_peak_mb", "std"),
            n_runs=("cg_gap_percent", "count"),
        ).reset_index()
        out_sum = os.path.join(out_dir, f"unified_single_{aircraft}_{tag}_summary.csv")
        agg.to_csv(out_sum, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_sum}")

    return out_csv


def run_demo(
    aircraft: str,
    loader: Any,
    problem_cls: Any,
    algos: List[type],
    flight_number: Optional[str],
    time_limit: Optional[int],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    if not flight_number:
        # pick the top-1 single flight
        split = loader.split_cargo_by_segment(n_single=1, n_multi=0)
        flights = split["single_flights"]
        if not flights:
            raise RuntimeError("No single flight available for demo.")
        flight_number = flights[0]

    print(f"\n[DEMO] aircraft={aircraft}, flight={flight_number}")

    flight_cargo = loader.get_flight_cargo(flight_number)
    if len(flight_cargo) == 0:
        raise RuntimeError("Demo flight has no cargo rows.")

    problem = problem_cls(
        cargo_holds=loader.cargo_holds,
        flight_params=loader.flight_params,
        cg_limits=loader.cg_limits,
        cargo_items=flight_cargo.reset_index(drop=True),
        segment_type="single",
    )

    hold_ids = [h["hold_id"] for h in problem.holds]

    tag = timestamp_str()
    demo_map_rows = []

    for algo_cls in algos:
        algo_name = getattr(algo_cls, "__name__", str(algo_cls))
        try:
            algo = instantiate_algorithm(algo_cls, problem, segment_type="single", time_limit=time_limit)
            result = run_algo(algo)
            sol = extract_solution(result, algo)
            sol_map = normalize_solution(sol, problem.n_items)

            print("\n" + "=" * 70)
            print(f"[DEMO] Algorithm: {getattr(algo, 'name', algo_name)}")
            print("=" * 70)

            # print cargo -> hold_id
            for item_idx in range(problem.n_items):
                hold_idx = sol_map.get(item_idx, -1)
                hold_id = "NOT_LOADED"
                if isinstance(hold_idx, (int, np.integer)) and 0 <= hold_idx < len(hold_ids):
                    hold_id = hold_ids[hold_idx]
                w = float(problem.cargo_items.iloc[item_idx]["weight"])
                dest = str(problem.cargo_items.iloc[item_idx].get("destination", ""))
                print(f"cargo[{item_idx:4d}] (w={w:.1f}, dest={dest}) -> hold={hold_id}")

                demo_map_rows.append({
                    "aircraft": aircraft,
                    "flight_number": flight_number,
                    "algo": getattr(algo, "name", algo_name),
                    "cargo_idx": item_idx,
                    "cargo_weight": w,
                    "destination": dest,
                    "hold_idx": hold_idx,
                    "hold_id": hold_id,
                })

        except Exception as e:
            print(f"[DEMO] {algo_name} Error: {e}")
            traceback.print_exc()

    if demo_map_rows:
        df_map = pd.DataFrame(demo_map_rows)
        out_csv = os.path.join(out_dir, f"demo_mapping_{aircraft}_{flight_number}_{tag}.csv")
        df_map.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[DEMO] mapping saved: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser("AirCa Unified Single+Demo Runner")

    p.add_argument("--code-root", type=str, default=DEFAULT_CODE_ROOT,
                   help="Project code root (must contain algorithm/ folder). Default is your original.")
    p.add_argument("--aircraft-data-dir", type=str, default=DEFAULT_AIRCRAFT_DATA_DIR,
                   help="Aircraft static data dir. Default moved to G:\\AirCa\\code\\aircraft_data")
    p.add_argument("--cargo-data-dir", type=str, default=DEFAULT_CARGO_DATA_DIR,
                   help="Cargo data dir containing BAKFLGITH_LOADDATA*.csv")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for combined CSVs.")
    p.add_argument("--aircraft", type=str, default="B777",
                   choices=["A320", "C919", "B777", "all"],
                   help="Which aircraft to run.")
    p.add_argument("--mode", type=str, default="demo",
                   choices=["single", "demo"],
                   help="single: batch single-segment; demo: pick one flight and print cargo->hold.")
    p.add_argument("--n-flights", type=int, default=100,
                   help="Number of single flights to test in single mode.")
    p.add_argument("--flight-number", type=str, default=None,
                   help="Demo flight number (optional). If omitted, auto-pick top single flight.")
    p.add_argument("--time-limit", type=int, default=None,
                   help="Optional time limit passed to algorithm constructors (if supported).")
    p.add_argument("--algos", type=str, default="all",
                   help="Comma-separated algorithm names to run, e.g. MILP,GA,PSO. Default all.")

    return p.parse_args()


def filter_algorithms(algos: List[type], spec: str) -> List[type]:
    if spec.strip().lower() == "all":
        return algos
    wanted = [x.strip().upper() for x in spec.split(",") if x.strip()]
    out = []
    for cls in algos:
        name = getattr(cls, "__name__", "").upper()
        if name in wanted:
            out.append(cls)
    return out if out else algos


def main():
    args = parse_args()
    ensure_sys_path(args.code_root)

    aircraft_list = ["A320", "C919", "B777"] if args.aircraft == "all" else [args.aircraft]

    # output subfolder to keep clean
    out_dir = os.path.join(args.output_dir, "unified_single_demo")
    os.makedirs(out_dir, exist_ok=True)

    for aircraft in aircraft_list:
        base_path = pick_base_path(args.aircraft_data_dir, aircraft)
        print("\n" + "#" * 80)
        print(f"Aircraft={aircraft}")
        print(f"BasePath={base_path}")
        print(f"CargoDataDir={args.cargo_data_dir}")
        print("#" * 80)

        loader, problem_cls = build_loader_and_problem(aircraft, base_path, args.cargo_data_dir)

        algos = load_algorithm_classes(aircraft)
        algos = filter_algorithms(algos, args.algos)

        if args.mode == "single":
            run_single(
                aircraft=aircraft,
                loader=loader,
                problem_cls=problem_cls,
                algos=algos,
                n_flights=args.n_flights,
                time_limit=args.time_limit,
                out_dir=out_dir,
            )
        else:
            run_demo(
                aircraft=aircraft,
                loader=loader,
                problem_cls=problem_cls,
                algos=algos,
                flight_number=args.flight_number,
                time_limit=args.time_limit,
                out_dir=out_dir,
            )


if __name__ == "__main__":
    main()
