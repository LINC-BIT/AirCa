#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cargo Loading Algorithms Package

支持两种机型：
1. 宽体机（B777等）：使用ULD，一个舱位只能装一个货物
2. 窄体机（A320等）：使用散装，一个舱位可以装多个货物

使用方法：
    # 宽体机
    algo = MILP(problem, segment_type='single', aircraft_type='widebody')

    # 窄体机
    algo = MILP(problem, segment_type='single', aircraft_type='narrowbody')
"""

# from .base_algorithm1 import BaseAlgorithm, ResultCollector
# from .milp_algorithm1 import MILP
# from .exact_algorithms1 import MINLP, QP, DP, CP
# from .heuristic_algorithms1 import GA, PSO, CS, ACO, ABC, MBO

from .base_algorithm import BaseAlgorithm, ResultCollector
# from .base_algorithm1 import BaseAlgorithm, ResultCollector
# from .milp_algorithm import MILP
from .exact_algorithms import MILP, MINLP, QP, DP, CP
from .heuristic_algorithms import GA, PSO, CS, ACO, ABC, MBO

__all__ = [
    'BaseAlgorithm',
    'ResultCollector',
    'MILP',
    'MINLP',
    'QP',
    'DP',
    'CP',
    'GA',
    'PSO',
    'CS',
    'ACO',
    'ABC',
    'MBO'
]
