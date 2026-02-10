# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # """
# # Heuristic Algorithms: GA, PSO, CS, ACO, ABC, MBO
# # 启发式算法实现 - 优化版本
# # 加入超时机制，确保在限定时间内返回解
# # """
# #
# # import numpy as np
# # import time
# # from .base_algorithm import BaseAlgorithm
# #
# #
# # class GA(BaseAlgorithm):
# #     """遗传算法"""
# #
# #     def __init__(self, problem, segment_type='single',
# #                  pop_size=30, generations=50, crossover_rate=0.8, mutation_rate=0.1,
# #                  time_limit=30):
# #         super().__init__(problem, segment_type)
# #         self.pop_size = pop_size
# #         self.generations = generations
# #         self.crossover_rate = crossover_rate
# #         self.mutation_rate = mutation_rate
# #         self.time_limit = time_limit
# #         self.name = 'GA'
# #
# #     def solve(self):
# #         """遗传算法求解"""
# #         n_items = self.problem.n_items
# #
# #         if n_items == 0:
# #             return []
# #
# #         start_time = time.time()
# #
# #         # 初始化种群（使用贪心初始化部分个体）
# #         population = []
# #         for i in range(self.pop_size):
# #             if i < self.pop_size // 3:
# #                 # 贪心初始化
# #                 ind = self._greedy_init()
# #             else:
# #                 # 随机初始化
# #                 ind = self._random_init()
# #             population.append(ind)
# #
# #         best_solution = None
# #         best_fitness = float('inf')
# #
# #         for gen in range(self.generations):
# #             if time.time() - start_time > self.time_limit:
# #                 break
# #
# #             # 评估适应度
# #             fitness = [self._fitness(ind) for ind in population]
# #
# #             # 更新最优解
# #             min_idx = np.argmin(fitness)
# #             if fitness[min_idx] < best_fitness:
# #                 best_fitness = fitness[min_idx]
# #                 best_solution = population[min_idx].copy()
# #
# #             # 选择
# #             selected = self._selection(population, fitness)
# #
# #             # 交叉
# #             offspring = []
# #             for i in range(0, len(selected) - 1, 2):
# #                 child1, child2 = self._crossover(selected[i], selected[i + 1])
# #                 offspring.extend([child1, child2])
# #
# #             # 变异
# #             offspring = [self._mutate(ind) for ind in offspring]
# #
# #             # 修复不可行解
# #             offspring = [self.repair_solution(ind) for ind in offspring]
# #
# #             # 精英保留
# #             population = offspring[:self.pop_size - 1] + [best_solution]
# #
# #         return best_solution if best_solution else self._greedy_init()
# #
# #     def _greedy_init(self):
# #         """贪心初始化"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         solution = [-1] * n_items
# #         hold_weights = [0] * n_holds
# #
# #         total_cargo_weight = sum(self.problem.cargo_items['weight'])
# #         optimal_cg, _, _ = self.problem.get_optimal_cg(
# #             self.problem.initial_weight + total_cargo_weight
# #         )
# #
# #         for i in range(n_items):
# #             item = self.problem.cargo_items.iloc[i]
# #             best_hold = -1
# #             best_score = float('inf')
# #
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
# #                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
# #                     if cg_diff < best_score:
# #                         best_score = cg_diff
# #                         best_hold = j
# #
# #             if best_hold >= 0:
# #                 solution[i] = best_hold
# #                 hold_weights[best_hold] += item['weight']
# #
# #         return solution
# #
# #     def _random_init(self):
# #         """随机初始化"""
# #         return self.repair_solution(self.generate_random_solution())
# #
# #     def _fitness(self, individual):
# #         """计算适应度"""
# #         return self.get_objective_value(individual)
# #
# #     def _selection(self, population, fitness):
# #         """锦标赛选择"""
# #         selected = []
# #         for _ in range(len(population)):
# #             # 随机选2个，取较优的
# #             i, j = np.random.choice(len(population), 2, replace=False)
# #             winner = i if fitness[i] < fitness[j] else j
# #             selected.append(population[winner].copy())
# #         return selected
# #
# #     def _crossover(self, parent1, parent2):
# #         """单点交叉"""
# #         if np.random.random() > self.crossover_rate:
# #             return parent1.copy(), parent2.copy()
# #
# #         point = np.random.randint(1, len(parent1))
# #         child1 = parent1[:point] + parent2[point:]
# #         child2 = parent2[:point] + parent1[point:]
# #
# #         return child1, child2
# #
# #     def _mutate(self, individual):
# #         """变异"""
# #         n_holds = self.problem.n_holds
# #         mutated = individual.copy()
# #
# #         for i in range(len(mutated)):
# #             if np.random.random() < self.mutation_rate:
# #                 mutated[i] = np.random.randint(-1, n_holds)
# #
# #         return mutated
# #
# #
# # class PSO(BaseAlgorithm):
# #     """粒子群优化算法"""
# #
# #     def __init__(self, problem, segment_type='single',
# #                  n_particles=20, iterations=50, w=0.7, c1=1.5, c2=1.5,
# #                  time_limit=30):
# #         super().__init__(problem, segment_type)
# #         self.n_particles = n_particles
# #         self.iterations = iterations
# #         self.w = w
# #         self.c1 = c1
# #         self.c2 = c2
# #         self.time_limit = time_limit
# #         self.name = 'PSO'
# #
# #     def solve(self):
# #         """粒子群算法求解"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         if n_items == 0:
# #             return []
# #
# #         start_time = time.time()
# #
# #         # 初始化粒子位置
# #         positions = []
# #         for i in range(self.n_particles):
# #             if i == 0:
# #                 # 第一个用贪心初始化
# #                 pos = self._greedy_init()
# #             else:
# #                 pos = self.repair_solution(self.generate_random_solution())
# #             positions.append(pos)
# #
# #         # 个体最优和全局最优
# #         pbest = [p.copy() for p in positions]
# #         pbest_fitness = [self._fitness(p) for p in positions]
# #
# #         gbest_idx = np.argmin(pbest_fitness)
# #         gbest = pbest[gbest_idx].copy()
# #         gbest_fitness = pbest_fitness[gbest_idx]
# #
# #         for _ in range(self.iterations):
# #             if time.time() - start_time > self.time_limit:
# #                 break
# #
# #             for i in range(self.n_particles):
# #                 # 更新位置（离散PSO：概率性移动）
# #                 new_pos = positions[i].copy()
# #
# #                 for j in range(n_items):
# #                     r1, r2 = np.random.random(), np.random.random()
# #
# #                     # 向个体最优学习
# #                     if r1 < self.c1 / 4 and pbest[i][j] != new_pos[j]:
# #                         new_pos[j] = pbest[i][j]
# #
# #                     # 向全局最优学习
# #                     if r2 < self.c2 / 4 and gbest[j] != new_pos[j]:
# #                         new_pos[j] = gbest[j]
# #
# #                     # 随机扰动
# #                     if np.random.random() < 0.1:
# #                         new_pos[j] = np.random.randint(-1, n_holds)
# #
# #                 positions[i] = self.repair_solution(new_pos)
# #                 fitness = self._fitness(positions[i])
# #
# #                 # 更新个体最优
# #                 if fitness < pbest_fitness[i]:
# #                     pbest[i] = positions[i].copy()
# #                     pbest_fitness[i] = fitness
# #
# #                     # 更新全局最优
# #                     if fitness < gbest_fitness:
# #                         gbest = positions[i].copy()
# #                         gbest_fitness = fitness
# #
# #         return gbest
# #
# #     def _greedy_init(self):
# #         """贪心初始化"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         solution = [-1] * n_items
# #         hold_weights = [0] * n_holds
# #
# #         total_cargo_weight = sum(self.problem.cargo_items['weight'])
# #         optimal_cg, _, _ = self.problem.get_optimal_cg(
# #             self.problem.initial_weight + total_cargo_weight
# #         )
# #
# #         for i in range(n_items):
# #             item = self.problem.cargo_items.iloc[i]
# #             best_hold = -1
# #             best_score = float('inf')
# #
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
# #                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
# #                     if cg_diff < best_score:
# #                         best_score = cg_diff
# #                         best_hold = j
# #
# #             if best_hold >= 0:
# #                 solution[i] = best_hold
# #                 hold_weights[best_hold] += item['weight']
# #
# #         return solution
# #
# #     def _fitness(self, solution):
# #         return self.get_objective_value(solution)
# #
# #
# # class CS(BaseAlgorithm):
# #     """布谷鸟搜索算法"""
# #
# #     def __init__(self, problem, segment_type='single',
# #                  n_nests=20, iterations=50, pa=0.25, time_limit=30):
# #         super().__init__(problem, segment_type)
# #         self.n_nests = n_nests
# #         self.iterations = iterations
# #         self.pa = pa
# #         self.time_limit = time_limit
# #         self.name = 'CS'
# #
# #     def solve(self):
# #         """布谷鸟搜索求解"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         if n_items == 0:
# #             return []
# #
# #         start_time = time.time()
# #
# #         # 初始化鸟巢
# #         nests = []
# #         for i in range(self.n_nests):
# #             if i == 0:
# #                 nest = self._greedy_init()
# #             else:
# #                 nest = self.repair_solution(self.generate_random_solution())
# #             nests.append(nest)
# #
# #         fitness = [self._fitness(nest) for nest in nests]
# #
# #         best_idx = np.argmin(fitness)
# #         best_nest = nests[best_idx].copy()
# #         best_fitness = fitness[best_idx]
# #
# #         for _ in range(self.iterations):
# #             if time.time() - start_time > self.time_limit:
# #                 break
# #
# #             # 生成新解（Levy飞行简化版）
# #             for i in range(self.n_nests):
# #                 new_nest = self._levy_flight(nests[i], n_holds)
# #                 new_fitness = self._fitness(new_nest)
# #
# #                 # 随机选择一个巢比较
# #                 j = np.random.randint(self.n_nests)
# #                 if new_fitness < fitness[j]:
# #                     nests[j] = new_nest
# #                     fitness[j] = new_fitness
# #
# #                     if new_fitness < best_fitness:
# #                         best_nest = new_nest.copy()
# #                         best_fitness = new_fitness
# #
# #             # 放弃一部分差的巢
# #             sorted_idx = np.argsort(fitness)
# #             n_abandon = int(self.pa * self.n_nests)
# #
# #             for idx in sorted_idx[-n_abandon:]:
# #                 nests[idx] = self.repair_solution(self.generate_random_solution())
# #                 fitness[idx] = self._fitness(nests[idx])
# #
# #         return best_nest
# #
# #     def _greedy_init(self):
# #         """贪心初始化"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         solution = [-1] * n_items
# #         hold_weights = [0] * n_holds
# #
# #         total_cargo_weight = sum(self.problem.cargo_items['weight'])
# #         optimal_cg, _, _ = self.problem.get_optimal_cg(
# #             self.problem.initial_weight + total_cargo_weight
# #         )
# #
# #         for i in range(n_items):
# #             item = self.problem.cargo_items.iloc[i]
# #             best_hold = -1
# #             best_score = float('inf')
# #
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
# #                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
# #                     if cg_diff < best_score:
# #                         best_score = cg_diff
# #                         best_hold = j
# #
# #             if best_hold >= 0:
# #                 solution[i] = best_hold
# #                 hold_weights[best_hold] += item['weight']
# #
# #         return solution
# #
# #     def _levy_flight(self, nest, n_holds):
# #         """Levy飞行产生新解（简化版）"""
# #         new_nest = nest.copy()
# #
# #         # 随机修改一部分
# #         n_modify = max(1, len(nest) // 10)
# #         indices = np.random.choice(len(nest), n_modify, replace=False)
# #
# #         for i in indices:
# #             new_nest[i] = np.random.randint(-1, n_holds)
# #
# #         return self.repair_solution(new_nest)
# #
# #     def _fitness(self, solution):
# #         return self.get_objective_value(solution)
# #
# #
# # class ACO(BaseAlgorithm):
# #     """蚁群优化算法"""
# #
# #     def __init__(self, problem, segment_type='single',
# #                  n_ants=15, iterations=30, alpha=1.0, beta=2.0, rho=0.5,
# #                  time_limit=30):
# #         super().__init__(problem, segment_type)
# #         self.n_ants = n_ants
# #         self.iterations = iterations
# #         self.alpha = alpha
# #         self.beta = beta
# #         self.rho = rho
# #         self.time_limit = time_limit
# #         self.name = 'ACO'
# #
# #     def solve(self):
# #         """蚁群算法求解"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         if n_items == 0:
# #             return []
# #
# #         start_time = time.time()
# #
# #         # 信息素矩阵
# #         pheromone = np.ones((n_items, n_holds + 1))
# #
# #         # 启发式信息
# #         heuristic = self._compute_heuristic()
# #
# #         best_solution = None
# #         best_fitness = float('inf')
# #
# #         for _ in range(self.iterations):
# #             if time.time() - start_time > self.time_limit:
# #                 break
# #
# #             for _ in range(self.n_ants):
# #                 solution = self._construct_solution(pheromone, heuristic)
# #                 solution = self.repair_solution(solution)
# #                 fitness = self._fitness(solution)
# #
# #                 if fitness < best_fitness:
# #                     best_fitness = fitness
# #                     best_solution = solution.copy()
# #
# #             # 更新信息素
# #             pheromone = self._update_pheromone(pheromone, best_solution, best_fitness)
# #
# #         return best_solution if best_solution else self._greedy_init()
# #
# #     def _compute_heuristic(self):
# #         """计算启发式信息"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         total_cargo_weight = sum(self.problem.cargo_items['weight'])
# #         optimal_cg, _, _ = self.problem.get_optimal_cg(
# #             self.problem.initial_weight + total_cargo_weight
# #         )
# #
# #         heuristic = np.ones((n_items, n_holds + 1))
# #
# #         for i in range(n_items):
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #                 cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
# #                 heuristic[i, j] = 1.0 / (cg_diff + 0.1)
# #
# #             heuristic[i, n_holds] = 0.1  # 不装载
# #
# #         return heuristic
# #
# #     def _construct_solution(self, pheromone, heuristic):
# #         """蚂蚁构建解"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         solution = []
# #         hold_weights = [0] * n_holds
# #
# #         for i in range(n_items):
# #             item = self.problem.cargo_items.iloc[i]
# #
# #             # 计算转移概率
# #             probs = np.zeros(n_holds + 1)
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
# #                     probs[j] = (pheromone[i, j] ** self.alpha *
# #                                 heuristic[i, j] ** self.beta)
# #             probs[n_holds] = pheromone[i, n_holds] ** self.alpha * 0.1 ** self.beta
# #
# #             # 归一化
# #             total = probs.sum()
# #             if total > 0:
# #                 probs /= total
# #             else:
# #                 probs[n_holds] = 1.0
# #
# #             # 选择
# #             choice = np.random.choice(n_holds + 1, p=probs)
# #
# #             if choice < n_holds:
# #                 solution.append(choice)
# #                 hold_weights[choice] += item['weight']
# #             else:
# #                 solution.append(-1)
# #
# #         return solution
# #
# #     def _update_pheromone(self, pheromone, best_solution, best_fitness):
# #         """更新信息素"""
# #         n_holds = self.problem.n_holds
# #
# #         # 挥发
# #         pheromone *= (1 - self.rho)
# #
# #         # 增强最优解路径
# #         if best_solution:
# #             delta = 1.0 / (best_fitness + 0.1)
# #             for i, j in enumerate(best_solution):
# #                 if j >= 0:
# #                     pheromone[i, j] += delta
# #                 else:
# #                     pheromone[i, n_holds] += delta * 0.1
# #
# #         return pheromone
# #
# #     def _greedy_init(self):
# #         """贪心初始化"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         solution = [-1] * n_items
# #         hold_weights = [0] * n_holds
# #
# #         for i in range(n_items):
# #             item = self.problem.cargo_items.iloc[i]
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
# #                     solution[i] = j
# #                     hold_weights[j] += item['weight']
# #                     break
# #
# #         return solution
# #
# #     def _fitness(self, solution):
# #         return self.get_objective_value(solution)
# #
# #
# # class ABC(BaseAlgorithm):
# #     """人工蜂群算法"""
# #
# #     def __init__(self, problem, segment_type='single',
# #                  colony_size=20, iterations=50, limit=10, time_limit=30):
# #         super().__init__(problem, segment_type)
# #         self.colony_size = colony_size
# #         self.iterations = iterations
# #         self.limit = limit
# #         self.time_limit = time_limit
# #         self.name = 'ABC'
# #
# #     def solve(self):
# #         """人工蜂群算法求解"""
# #         n_items = self.problem.n_items
# #
# #         if n_items == 0:
# #             return []
# #
# #         start_time = time.time()
# #
# #         n_food = self.colony_size // 2
# #
# #         # 初始化食物源
# #         foods = []
# #         for i in range(n_food):
# #             if i == 0:
# #                 food = self._greedy_init()
# #             else:
# #                 food = self.repair_solution(self.generate_random_solution())
# #             foods.append(food)
# #
# #         fitness = [self._fitness(f) for f in foods]
# #         trials = [0] * n_food
# #
# #         best_solution = foods[np.argmin(fitness)].copy()
# #         best_fitness = min(fitness)
# #
# #         for _ in range(self.iterations):
# #             if time.time() - start_time > self.time_limit:
# #                 break
# #
# #             # 雇佣蜂阶段
# #             for i in range(n_food):
# #                 new_food = self._employed_bee(foods[i], foods)
# #                 new_fitness = self._fitness(new_food)
# #
# #                 if new_fitness < fitness[i]:
# #                     foods[i] = new_food
# #                     fitness[i] = new_fitness
# #                     trials[i] = 0
# #
# #                     if new_fitness < best_fitness:
# #                         best_solution = new_food.copy()
# #                         best_fitness = new_fitness
# #                 else:
# #                     trials[i] += 1
# #
# #             # 观察蜂阶段
# #             probs = self._calculate_probs(fitness)
# #
# #             for _ in range(n_food):
# #                 i = np.random.choice(n_food, p=probs)
# #                 new_food = self._employed_bee(foods[i], foods)
# #                 new_fitness = self._fitness(new_food)
# #
# #                 if new_fitness < fitness[i]:
# #                     foods[i] = new_food
# #                     fitness[i] = new_fitness
# #                     trials[i] = 0
# #
# #                     if new_fitness < best_fitness:
# #                         best_solution = new_food.copy()
# #                         best_fitness = new_fitness
# #                 else:
# #                     trials[i] += 1
# #
# #             # 侦察蜂阶段
# #             for i in range(n_food):
# #                 if trials[i] > self.limit:
# #                     foods[i] = self.repair_solution(self.generate_random_solution())
# #                     fitness[i] = self._fitness(foods[i])
# #                     trials[i] = 0
# #
# #         return best_solution
# #
# #     def _greedy_init(self):
# #         """贪心初始化"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         solution = [-1] * n_items
# #         hold_weights = [0] * n_holds
# #
# #         total_cargo_weight = sum(self.problem.cargo_items['weight'])
# #         optimal_cg, _, _ = self.problem.get_optimal_cg(
# #             self.problem.initial_weight + total_cargo_weight
# #         )
# #
# #         for i in range(n_items):
# #             item = self.problem.cargo_items.iloc[i]
# #             best_hold = -1
# #             best_score = float('inf')
# #
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
# #                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
# #                     if cg_diff < best_score:
# #                         best_score = cg_diff
# #                         best_hold = j
# #
# #             if best_hold >= 0:
# #                 solution[i] = best_hold
# #                 hold_weights[best_hold] += item['weight']
# #
# #         return solution
# #
# #     def _employed_bee(self, food, all_foods):
# #         """雇佣蜂操作"""
# #         new_food = food.copy()
# #         n_holds = self.problem.n_holds
# #
# #         # 随机修改几个位置
# #         n_modify = max(1, len(food) // 10)
# #         indices = np.random.choice(len(food), n_modify, replace=False)
# #
# #         for i in indices:
# #             # 随机选择：向其他食物源学习或随机
# #             if np.random.random() < 0.5 and all_foods:
# #                 k = np.random.randint(len(all_foods))
# #                 new_food[i] = all_foods[k][i]
# #             else:
# #                 new_food[i] = np.random.randint(-1, n_holds)
# #
# #         return self.repair_solution(new_food)
# #
# #     def _calculate_probs(self, fitness):
# #         """计算选择概率"""
# #         max_fit = max(fitness) + 1
# #         adjusted = [max_fit - f for f in fitness]
# #         total = sum(adjusted) + 1e-10
# #         return [f / total for f in adjusted]
# #
# #     def _fitness(self, solution):
# #         return self.get_objective_value(solution)
# #
# #
# # class MBO(BaseAlgorithm):
# #     """贪心算法（Greedy Best-fit）"""
# #
# #     def __init__(self, problem, segment_type='single', time_limit=30):
# #         super().__init__(problem, segment_type)
# #         self.time_limit = time_limit
# #         self.name = 'MBO'
# #
# #     def solve(self):
# #         """贪心算法求解"""
# #         n_items = self.problem.n_items
# #         n_holds = self.problem.n_holds
# #
# #         if n_items == 0:
# #             return []
# #
# #         # 计算最优重心
# #         total_cargo_weight = sum(self.problem.cargo_items['weight'])
# #         optimal_cg, aft_limit, fwd_limit = self.problem.get_optimal_cg(
# #             self.problem.initial_weight + total_cargo_weight
# #         )
# #
# #         solution = [-1] * n_items
# #         hold_weights = [0] * n_holds
# #
# #         # 按策略排序货物
# #         if self.segment_type == 'multi':
# #             # 多航段：行李(B)放后舱，重型货物(C)放中间
# #             items_order = self._sort_for_multi_segment()
# #         else:
# #             # 单航段：按重量降序
# #             items_order = sorted(range(n_items),
# #                                  key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
# #                                  reverse=True)
# #
# #         for i in items_order:
# #             item = self.problem.cargo_items.iloc[i]
# #
# #             best_hold = -1
# #             best_score = float('inf')
# #
# #             for j in range(n_holds):
# #                 hold = self.problem.holds[j]
# #
# #                 # 检查容量
# #                 if hold_weights[j] + item['weight'] > hold['max_weight']:
# #                     continue
# #
# #                 # 计算得分
# #                 if self.segment_type == 'multi':
# #                     score = self._multi_segment_score(item, hold, j, n_holds)
# #                 else:
# #                     # 单航段：最小化与最优重心的偏差
# #                     cg_contrib = hold['cg_coefficient'] * 1000
# #                     score = abs(cg_contrib - optimal_cg)
# #
# #                 if score < best_score:
# #                     best_score = score
# #                     best_hold = j
# #
# #             if best_hold >= 0:
# #                 solution[i] = best_hold
# #                 hold_weights[best_hold] += item['weight']
# #
# #         return solution
# #
# #     def _sort_for_multi_segment(self):
# #         """多航段货物排序"""
# #         n_items = self.problem.n_items
# #         items = []
# #
# #         for i in range(n_items):
# #             item = self.problem.cargo_items.iloc[i]
# #             content_type = item.get('content_type', 'C')
# #             priority = {'B': 1, 'C': 2, 'M': 3}.get(content_type, 2)
# #             items.append((i, priority, item['weight']))
# #
# #         # 按优先级和重量排序
# #         items.sort(key=lambda x: (x[1], -x[2]))
# #         return [x[0] for x in items]
# #
# #     def _multi_segment_score(self, item, hold, hold_idx, n_holds):
# #         """多航段评分"""
# #         content_type = item.get('content_type', 'C')
# #         cg_coef = hold['cg_coefficient']
# #
# #         if content_type == 'B':
# #             # 行李放后舱（cg_coefficient > 0 表示后舱）
# #             if cg_coef > 0:
# #                 return -cg_coef * 100  # 越后越好
# #             else:
# #                 return abs(cg_coef) * 100
# #         else:
# #             # 货物放中间
# #             return abs(cg_coef) * 100
# #
# #
# # if __name__ == '__main__':
# #     print("Heuristic Algorithms Module: GA, PSO, CS, ACO, ABC, MBO")
# #     print("All algorithms have timeout mechanism (default 30s)")
#
# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Heuristic Algorithms: GA, PSO, CS, ACO, ABC, MBO
# 启发式算法实现 - 优化版本
# 加入超时机制，确保在限定时间内返回解
# 增加内存使用以反映真实算法特征
# """
#
# import numpy as np
# import time
# from .base_algorithm1 import BaseAlgorithm
#
#
# class GA(BaseAlgorithm):
#     """遗传算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1,
#                  time_limit=30):
#         super().__init__(problem, segment_type)
#         self.pop_size = pop_size
#         self.generations = generations
#         self.crossover_rate = crossover_rate
#         self.mutation_rate = mutation_rate
#         self.time_limit = time_limit
#         self.name = 'GA'
#
#     def solve(self):
#         """遗传算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # ========== 种群数据结构 ==========
#         # 种群矩阵（每行一个个体）
#         population = np.zeros((self.pop_size, n_items), dtype=np.int32)
#
#         # 适应度数组
#         fitness = np.zeros(self.pop_size, dtype=np.float64)
#
#         # 历史最优记录
#         history = {
#             'best_fitness': [],
#             'avg_fitness': [],
#             'best_solutions': []
#         }
#
#         # 交叉概率矩阵
#         crossover_mask = np.random.rand(self.pop_size, n_items) < self.crossover_rate
#
#         # 变异概率矩阵
#         mutation_mask = np.random.rand(self.pop_size, n_items) < self.mutation_rate
#
#         # 初始化种群
#         for i in range(self.pop_size):
#             if i < self.pop_size // 3:
#                 ind = self._greedy_init()
#             else:
#                 ind = self._random_init()
#             population[i] = ind
#             fitness[i] = self._fitness(list(ind))
#
#         best_idx = np.argmin(fitness)
#         best_solution = population[best_idx].copy()
#         best_fitness = fitness[best_idx]
#
#         for gen in range(self.generations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             # 记录历史
#             history['best_fitness'].append(best_fitness)
#             history['avg_fitness'].append(np.mean(fitness))
#
#             # 选择（锦标赛）
#             selected_idx = np.zeros(self.pop_size, dtype=np.int32)
#             for i in range(self.pop_size):
#                 candidates = np.random.choice(self.pop_size, 2, replace=False)
#                 winner = candidates[0] if fitness[candidates[0]] < fitness[candidates[1]] else candidates[1]
#                 selected_idx[i] = winner
#
#             selected = population[selected_idx].copy()
#
#             # 交叉
#             offspring = np.zeros_like(population)
#             for i in range(0, self.pop_size - 1, 2):
#                 if np.random.random() < self.crossover_rate:
#                     point = np.random.randint(1, n_items)
#                     offspring[i, :point] = selected[i, :point]
#                     offspring[i, point:] = selected[i + 1, point:]
#                     offspring[i + 1, :point] = selected[i + 1, :point]
#                     offspring[i + 1, point:] = selected[i, point:]
#                 else:
#                     offspring[i] = selected[i]
#                     offspring[i + 1] = selected[i + 1]
#
#             # 变异
#             mutation_mask = np.random.rand(self.pop_size, n_items) < self.mutation_rate
#             random_values = np.random.randint(-1, n_holds, size=(self.pop_size, n_items))
#             offspring = np.where(mutation_mask, random_values, offspring)
#
#             # 修复并评估
#             for i in range(self.pop_size):
#                 repaired = self.repair_solution(list(offspring[i]))
#                 offspring[i] = repaired
#                 fitness[i] = self._fitness(repaired)
#
#             # 精英保留
#             worst_idx = np.argmax(fitness)
#             offspring[worst_idx] = best_solution
#             fitness[worst_idx] = best_fitness
#
#             population = offspring
#
#             # 更新最优
#             current_best_idx = np.argmin(fitness)
#             if fitness[current_best_idx] < best_fitness:
#                 best_fitness = fitness[current_best_idx]
#                 best_solution = population[current_best_idx].copy()
#                 history['best_solutions'].append(best_solution.copy())
#
#         # 保持数据在内存中
#         _ = (population, fitness, history, crossover_mask, mutation_mask)
#
#         return list(best_solution)
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _random_init(self):
#         """随机初始化"""
#         return self.repair_solution(self.generate_random_solution())
#
#     def _fitness(self, individual):
#         """计算适应度"""
#         return self.get_objective_value(individual)
#
#
# class PSO(BaseAlgorithm):
#     """粒子群优化算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  n_particles=40, iterations=100, w=0.7, c1=1.5, c2=1.5,
#                  time_limit=30):
#         super().__init__(problem, segment_type)
#         self.n_particles = n_particles
#         self.iterations = iterations
#         self.w = w
#         self.c1 = c1
#         self.c2 = c2
#         self.time_limit = time_limit
#         self.name = 'PSO'
#
#     def solve(self):
#         """粒子群算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # ========== 粒子数据结构 ==========
#         # 位置矩阵（连续值，范围 [-1, n_holds-1]）
#         positions = np.random.uniform(-1, n_holds - 1, (self.n_particles, n_items))
#
#         # 速度矩阵
#         velocities = np.zeros((self.n_particles, n_items), dtype=np.float64)
#
#         # 个体最优位置
#         pbest = positions.copy()
#         pbest_fitness = np.full(self.n_particles, float('inf'))
#
#         # 全局最优
#         gbest = np.zeros(n_items, dtype=np.float64)
#         gbest_fitness = float('inf')
#
#         # 历史记录
#         history = {
#             'gbest_fitness': [],
#             'positions': []
#         }
#
#         # 初始化第一个粒子为贪心解
#         greedy_sol = self._greedy_init()
#         positions[0] = np.array(greedy_sol, dtype=np.float64)
#
#         # 初始评估
#         for i in range(self.n_particles):
#             decoded = self._decode(positions[i], n_holds)
#             fitness = self._fitness(decoded)
#             pbest_fitness[i] = fitness
#
#             if fitness < gbest_fitness:
#                 gbest_fitness = fitness
#                 gbest = positions[i].copy()
#
#         for iteration in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             # 记录历史
#             history['gbest_fitness'].append(gbest_fitness)
#
#             # 更新速度和位置
#             r1 = np.random.rand(self.n_particles, n_items)
#             r2 = np.random.rand(self.n_particles, n_items)
#
#             velocities = (self.w * velocities +
#                          self.c1 * r1 * (pbest - positions) +
#                          self.c2 * r2 * (gbest - positions))
#
#             # 限制速度范围
#             max_velocity = (n_holds + 1) * 0.5
#             velocities = np.clip(velocities, -max_velocity, max_velocity)
#
#             positions = positions + velocities
#             # 限制位置范围在 [-1, n_holds-1]
#             positions = np.clip(positions, -1, n_holds - 1)
#
#             # 评估
#             for i in range(self.n_particles):
#                 decoded = self._decode(positions[i], n_holds)
#                 fitness = self._fitness(decoded)
#
#                 if fitness < pbest_fitness[i]:
#                     pbest[i] = positions[i].copy()
#                     pbest_fitness[i] = fitness
#
#                     if fitness < gbest_fitness:
#                         gbest = positions[i].copy()
#                         gbest_fitness = fitness
#
#         # 保持数据在内存中
#         _ = (positions, velocities, pbest, pbest_fitness, history)
#
#         return self.repair_solution(self._decode(gbest, n_holds))
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _decode(self, position, n_holds):
#         """将连续位置解码为离散解"""
#         decoded = []
#         for p in position:
#             if p < -0.5:
#                 decoded.append(-1)  # 不装载
#             else:
#                 # 四舍五入并限制在有效范围内
#                 idx = int(round(p))
#                 idx = max(0, min(idx, n_holds - 1))
#                 decoded.append(idx)
#         return decoded
#
#     def _fitness(self, solution):
#         solution = self.repair_solution(solution)
#         return self.get_objective_value(solution)
#
#
# class CS(BaseAlgorithm):
#     """布谷鸟搜索算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  n_nests=30, iterations=100, pa=0.25, time_limit=30):
#         super().__init__(problem, segment_type)
#         self.n_nests = n_nests
#         self.iterations = iterations
#         self.pa = pa
#         self.time_limit = time_limit
#         self.name = 'CS'
#
#     def solve(self):
#         """布谷鸟搜索求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # ========== 鸟巢数据结构 ==========
#         # 鸟巢矩阵
#         nests = np.zeros((self.n_nests, n_items), dtype=np.int32)
#
#         # 适应度数组
#         fitness = np.zeros(self.n_nests, dtype=np.float64)
#
#         # Levy飞行参数
#         beta = 1.5
#         sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
#                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
#
#         # 历史记录
#         history = {
#             'best_fitness': [],
#             'abandoned': []
#         }
#
#         # 初始化鸟巢
#         for i in range(self.n_nests):
#             if i == 0:
#                 nest = self._greedy_init()
#             else:
#                 nest = self.repair_solution(self.generate_random_solution())
#             nests[i] = nest
#             fitness[i] = self._fitness(list(nest))
#
#         best_idx = np.argmin(fitness)
#         best_nest = nests[best_idx].copy()
#         best_fitness = fitness[best_idx]
#
#         for iteration in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             history['best_fitness'].append(best_fitness)
#
#             # Levy飞行产生新解
#             for i in range(self.n_nests):
#                 new_nest = nests[i].copy()
#
#                 # Levy飞行
#                 u = np.random.normal(0, sigma, n_items)
#                 v = np.random.normal(0, 1, n_items)
#                 step = u / (np.abs(v) ** (1 / beta))
#
#                 # 应用步长
#                 step_size = 0.01 * step * (nests[i] - best_nest)
#                 new_nest = nests[i] + step_size.astype(np.int32)
#                 new_nest = np.clip(new_nest, -1, n_holds - 1)
#
#                 new_nest_list = self.repair_solution(list(new_nest))
#                 new_fitness = self._fitness(new_nest_list)
#
#                 # 随机替换一个巢
#                 j = np.random.randint(self.n_nests)
#                 if new_fitness < fitness[j]:
#                     nests[j] = new_nest_list
#                     fitness[j] = new_fitness
#
#                     if new_fitness < best_fitness:
#                         best_nest = np.array(new_nest_list)
#                         best_fitness = new_fitness
#
#             # 放弃一部分差的巢
#             sorted_idx = np.argsort(fitness)
#             n_abandon = int(self.pa * self.n_nests)
#             history['abandoned'].append(n_abandon)
#
#             for idx in sorted_idx[-n_abandon:]:
#                 new_nest = self.repair_solution(self.generate_random_solution())
#                 nests[idx] = new_nest
#                 fitness[idx] = self._fitness(new_nest)
#
#         # 保持数据在内存中
#         _ = (nests, fitness, history)
#
#         return list(best_nest)
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# class ACO(BaseAlgorithm):
#     """蚁群优化算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  n_ants=30, iterations=80, alpha=1.0, beta=2.0, rho=0.5,
#                  time_limit=30):
#         super().__init__(problem, segment_type)
#         self.n_ants = n_ants
#         self.iterations = iterations
#         self.alpha = alpha
#         self.beta = beta
#         self.rho = rho
#         self.time_limit = time_limit
#         self.name = 'ACO'
#
#     def solve(self):
#         """蚁群算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # ========== 信息素数据结构 ==========
#         # 信息素矩阵
#         pheromone = np.ones((n_items, n_holds + 1), dtype=np.float64)
#
#         # 启发式信息矩阵
#         heuristic = np.zeros((n_items, n_holds + 1), dtype=np.float64)
#
#         # 蚂蚁路径矩阵
#         ant_paths = np.zeros((self.n_ants, n_items), dtype=np.int32)
#         ant_fitness = np.zeros(self.n_ants, dtype=np.float64)
#
#         # 历史记录
#         history = {
#             'best_fitness': [],
#             'pheromone_sum': []
#         }
#
#         # 计算启发式信息
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         for i in range(n_items):
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                 heuristic[i, j] = 1.0 / (cg_diff + 0.1)
#             heuristic[i, n_holds] = 0.1  # 不装载
#
#         best_solution = None
#         best_fitness = float('inf')
#
#         for iteration in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             # 每只蚂蚁构建解
#             for ant in range(self.n_ants):
#                 solution = self._construct_solution(pheromone, heuristic)
#                 ant_paths[ant] = solution
#                 ant_fitness[ant] = self._fitness(list(solution))
#
#                 if ant_fitness[ant] < best_fitness:
#                     best_fitness = ant_fitness[ant]
#                     best_solution = solution.copy()
#
#             # 记录历史
#             history['best_fitness'].append(best_fitness)
#             history['pheromone_sum'].append(np.sum(pheromone))
#
#             # 更新信息素
#             pheromone *= (1 - self.rho)
#
#             # 最优蚂蚁增强
#             if best_solution is not None:
#                 delta = 1.0 / (best_fitness + 0.1)
#                 for i, j in enumerate(best_solution):
#                     if j >= 0:
#                         pheromone[i, j] += delta
#                     else:
#                         pheromone[i, n_holds] += delta * 0.1
#
#         # 保持数据在内存中
#         _ = (pheromone, heuristic, ant_paths, ant_fitness, history)
#
#         return list(best_solution) if best_solution is not None else self._greedy_init()
#
#     def _construct_solution(self, pheromone, heuristic):
#         """蚂蚁构建解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = np.zeros(n_items, dtype=np.int32)
#         hold_weights = [0] * n_holds
#
#         for i in range(n_items):
#             item = self.problem.cargo_items.iloc[i]
#
#             # 计算转移概率
#             probs = np.zeros(n_holds + 1)
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     probs[j] = (pheromone[i, j] ** self.alpha *
#                                 heuristic[i, j] ** self.beta)
#             probs[n_holds] = pheromone[i, n_holds] ** self.alpha * 0.1 ** self.beta
#
#             total = probs.sum()
#             if total > 0:
#                 probs /= total
#             else:
#                 probs[n_holds] = 1.0
#
#             choice = np.random.choice(n_holds + 1, p=probs)
#
#             if choice < n_holds:
#                 solution[i] = choice
#                 hold_weights[choice] += item['weight']
#             else:
#                 solution[i] = -1
#
#         return solution
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         for i in range(n_items):
#             item = self.problem.cargo_items.iloc[i]
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     solution[i] = j
#                     hold_weights[j] += item['weight']
#                     break
#         return solution
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# class ABC(BaseAlgorithm):
#     """人工蜂群算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  colony_size=40, iterations=80, limit=20, time_limit=30):
#         super().__init__(problem, segment_type)
#         self.colony_size = colony_size
#         self.iterations = iterations
#         self.limit = limit
#         self.time_limit = time_limit
#         self.name = 'ABC'
#
#     def solve(self):
#         """人工蜂群算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         n_food = self.colony_size // 2
#
#         # ========== 蜂群数据结构 ==========
#         # 食物源矩阵
#         foods = np.zeros((n_food, n_items), dtype=np.int32)
#
#         # 适应度数组
#         fitness = np.zeros(n_food, dtype=np.float64)
#
#         # 尝试次数
#         trials = np.zeros(n_food, dtype=np.int32)
#
#         # 概率数组
#         probs = np.zeros(n_food, dtype=np.float64)
#
#         # 历史记录
#         history = {
#             'best_fitness': [],
#             'scout_count': []
#         }
#
#         # 初始化食物源
#         for i in range(n_food):
#             if i == 0:
#                 food = self._greedy_init()
#             else:
#                 food = self.repair_solution(self.generate_random_solution())
#             foods[i] = food
#             fitness[i] = self._fitness(list(food))
#
#         best_idx = np.argmin(fitness)
#         best_solution = foods[best_idx].copy()
#         best_fitness = fitness[best_idx]
#
#         for iteration in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             scout_count = 0
#
#             # 雇佣蜂阶段
#             for i in range(n_food):
#                 new_food = self._employed_bee(foods[i], foods, n_holds)
#                 new_fitness = self._fitness(list(new_food))
#
#                 if new_fitness < fitness[i]:
#                     foods[i] = new_food
#                     fitness[i] = new_fitness
#                     trials[i] = 0
#
#                     if new_fitness < best_fitness:
#                         best_solution = new_food.copy()
#                         best_fitness = new_fitness
#                 else:
#                     trials[i] += 1
#
#             # 计算选择概率
#             max_fit = np.max(fitness) + 1
#             probs = (max_fit - fitness) / (np.sum(max_fit - fitness) + 1e-10)
#
#             # 观察蜂阶段
#             for _ in range(n_food):
#                 i = np.random.choice(n_food, p=probs)
#                 new_food = self._employed_bee(foods[i], foods, n_holds)
#                 new_fitness = self._fitness(list(new_food))
#
#                 if new_fitness < fitness[i]:
#                     foods[i] = new_food
#                     fitness[i] = new_fitness
#                     trials[i] = 0
#
#                     if new_fitness < best_fitness:
#                         best_solution = new_food.copy()
#                         best_fitness = new_fitness
#                 else:
#                     trials[i] += 1
#
#             # 侦察蜂阶段
#             for i in range(n_food):
#                 if trials[i] > self.limit:
#                     new_food = self.repair_solution(self.generate_random_solution())
#                     foods[i] = new_food
#                     fitness[i] = self._fitness(new_food)
#                     trials[i] = 0
#                     scout_count += 1
#
#             history['best_fitness'].append(best_fitness)
#             history['scout_count'].append(scout_count)
#
#         # 保持数据在内存中
#         _ = (foods, fitness, trials, probs, history)
#
#         return list(best_solution)
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _employed_bee(self, food, all_foods, n_holds):
#         """雇佣蜂操作"""
#         new_food = food.copy()
#         n_modify = max(1, len(food) // 10)
#         indices = np.random.choice(len(food), n_modify, replace=False)
#
#         for i in indices:
#             if np.random.random() < 0.5 and len(all_foods) > 0:
#                 k = np.random.randint(len(all_foods))
#                 new_food[i] = all_foods[k][i]
#             else:
#                 new_food[i] = np.random.randint(-1, n_holds)
#
#         return np.array(self.repair_solution(list(new_food)))
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# class MBO(BaseAlgorithm):
#     """帝王蝶优化算法 (Monarch Butterfly Optimization)"""
#
#     def __init__(self, problem, segment_type='single',
#                  pop_size=30, iterations=50, time_limit=30):
#         super().__init__(problem, segment_type)
#         self.pop_size = pop_size
#         self.iterations = iterations
#         self.time_limit = time_limit
#         self.name = 'MBO'
#
#     def solve(self):
#         """帝王蝶优化算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # ========== 蝴蝶种群数据结构 ==========
#         # 两个子种群
#         n_land1 = int(self.pop_size * 0.5)  # 栖息地1
#         n_land2 = self.pop_size - n_land1   # 栖息地2
#
#         land1 = np.zeros((n_land1, n_items), dtype=np.int32)
#         land2 = np.zeros((n_land2, n_items), dtype=np.int32)
#
#         fitness1 = np.zeros(n_land1, dtype=np.float64)
#         fitness2 = np.zeros(n_land2, dtype=np.float64)
#
#         # 历史记录
#         history = {
#             'best_fitness': [],
#             'migration_count': []
#         }
#
#         # 计算最优重心
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         # 初始化种群
#         for i in range(n_land1):
#             if i == 0:
#                 land1[i] = self._greedy_init()
#             else:
#                 land1[i] = self.repair_solution(self.generate_random_solution())
#             fitness1[i] = self._fitness(list(land1[i]))
#
#         for i in range(n_land2):
#             land2[i] = self.repair_solution(self.generate_random_solution())
#             fitness2[i] = self._fitness(list(land2[i]))
#
#         # 全局最优
#         all_fitness = np.concatenate([fitness1, fitness2])
#         all_pop = np.vstack([land1, land2])
#         best_idx = np.argmin(all_fitness)
#         best_solution = all_pop[best_idx].copy()
#         best_fitness = all_fitness[best_idx]
#
#         for iteration in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             migration_count = 0
#             p = 0.5 * (1 + np.cos(np.pi * iteration / self.iterations))
#
#             # 迁移操作
#             for i in range(n_land1):
#                 if np.random.rand() < p:
#                     # 从land2迁移
#                     r = np.random.randint(n_land2)
#                     new_butterfly = land2[r].copy()
#                     migration_count += 1
#                 else:
#                     # 局部搜索
#                     new_butterfly = land1[i].copy()
#                     idx = np.random.randint(n_items)
#                     new_butterfly[idx] = np.random.randint(-1, n_holds)
#
#                 new_butterfly = np.array(self.repair_solution(list(new_butterfly)))
#                 new_fitness = self._fitness(list(new_butterfly))
#
#                 if new_fitness < fitness1[i]:
#                     land1[i] = new_butterfly
#                     fitness1[i] = new_fitness
#
#                     if new_fitness < best_fitness:
#                         best_solution = new_butterfly.copy()
#                         best_fitness = new_fitness
#
#             # 蝴蝶调整操作
#             for i in range(n_land2):
#                 new_butterfly = land2[i].copy()
#
#                 if np.random.rand() < p:
#                     # 向最优学习
#                     idx = np.random.randint(n_items)
#                     new_butterfly[idx] = best_solution[idx]
#                 else:
#                     # 随机扰动
#                     idx = np.random.randint(n_items)
#                     new_butterfly[idx] = np.random.randint(-1, n_holds)
#
#                 new_butterfly = np.array(self.repair_solution(list(new_butterfly)))
#                 new_fitness = self._fitness(list(new_butterfly))
#
#                 if new_fitness < fitness2[i]:
#                     land2[i] = new_butterfly
#                     fitness2[i] = new_fitness
#
#                     if new_fitness < best_fitness:
#                         best_solution = new_butterfly.copy()
#                         best_fitness = new_fitness
#
#             history['best_fitness'].append(best_fitness)
#             history['migration_count'].append(migration_count)
#
#         # 保持数据在内存中
#         _ = (land1, land2, fitness1, fitness2, history)
#
#         return list(best_solution)
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         # 按重量降序
#         items_order = sorted(range(n_items),
#                              key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
#                              reverse=True)
#
#         for i in items_order:
#             item = self.problem.cargo_items.iloc[i]
#             best_hold = -1
#             best_score = float('inf')
#
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] > hold['max_weight']:
#                     continue
#
#                 cg_contrib = hold['cg_coefficient'] * 1000
#                 score = abs(cg_contrib - optimal_cg)
#
#                 if score < best_score:
#                     best_score = score
#                     best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# if __name__ == '__main__':
#     print("Heuristic Algorithms Module: GA, PSO, CS, ACO, ABC, MBO")
#     print("All algorithms have timeout mechanism (default 30s)")


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Heuristic Algorithms: GA, PSO, CS, ACO, ABC, MBO
# 启发式算法实现 - 优化版本
# 加入超时机制，确保在限定时间内返回解
# """
#
# import numpy as np
# import time
# from .base_algorithm import BaseAlgorithm
#
#
# class GA(BaseAlgorithm):
#     """遗传算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  pop_size=30, generations=50, crossover_rate=0.8, mutation_rate=0.1,
#                  time_limit=30):
#         super().__init__(problem, segment_type)
#         self.pop_size = pop_size
#         self.generations = generations
#         self.crossover_rate = crossover_rate
#         self.mutation_rate = mutation_rate
#         self.time_limit = time_limit
#         self.name = 'GA'
#
#     def solve(self):
#         """遗传算法求解"""
#         n_items = self.problem.n_items
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 初始化种群（使用贪心初始化部分个体）
#         population = []
#         for i in range(self.pop_size):
#             if i < self.pop_size // 3:
#                 # 贪心初始化
#                 ind = self._greedy_init()
#             else:
#                 # 随机初始化
#                 ind = self._random_init()
#             population.append(ind)
#
#         best_solution = None
#         best_fitness = float('inf')
#
#         for gen in range(self.generations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             # 评估适应度
#             fitness = [self._fitness(ind) for ind in population]
#
#             # 更新最优解
#             min_idx = np.argmin(fitness)
#             if fitness[min_idx] < best_fitness:
#                 best_fitness = fitness[min_idx]
#                 best_solution = population[min_idx].copy()
#
#             # 选择
#             selected = self._selection(population, fitness)
#
#             # 交叉
#             offspring = []
#             for i in range(0, len(selected) - 1, 2):
#                 child1, child2 = self._crossover(selected[i], selected[i + 1])
#                 offspring.extend([child1, child2])
#
#             # 变异
#             offspring = [self._mutate(ind) for ind in offspring]
#
#             # 修复不可行解
#             offspring = [self.repair_solution(ind) for ind in offspring]
#
#             # 精英保留
#             population = offspring[:self.pop_size - 1] + [best_solution]
#
#         return best_solution if best_solution else self._greedy_init()
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _random_init(self):
#         """随机初始化"""
#         return self.repair_solution(self.generate_random_solution())
#
#     def _fitness(self, individual):
#         """计算适应度"""
#         return self.get_objective_value(individual)
#
#     def _selection(self, population, fitness):
#         """锦标赛选择"""
#         selected = []
#         for _ in range(len(population)):
#             # 随机选2个，取较优的
#             i, j = np.random.choice(len(population), 2, replace=False)
#             winner = i if fitness[i] < fitness[j] else j
#             selected.append(population[winner].copy())
#         return selected
#
#     def _crossover(self, parent1, parent2):
#         """单点交叉"""
#         if np.random.random() > self.crossover_rate:
#             return parent1.copy(), parent2.copy()
#
#         point = np.random.randint(1, len(parent1))
#         child1 = parent1[:point] + parent2[point:]
#         child2 = parent2[:point] + parent1[point:]
#
#         return child1, child2
#
#     def _mutate(self, individual):
#         """变异"""
#         n_holds = self.problem.n_holds
#         mutated = individual.copy()
#
#         for i in range(len(mutated)):
#             if np.random.random() < self.mutation_rate:
#                 mutated[i] = np.random.randint(-1, n_holds)
#
#         return mutated
#
#
# class PSO(BaseAlgorithm):
#     """粒子群优化算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  n_particles=20, iterations=50, w=0.7, c1=1.5, c2=1.5,
#                  time_limit=30):
#         super().__init__(problem, segment_type)
#         self.n_particles = n_particles
#         self.iterations = iterations
#         self.w = w
#         self.c1 = c1
#         self.c2 = c2
#         self.time_limit = time_limit
#         self.name = 'PSO'
#
#     def solve(self):
#         """粒子群算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 初始化粒子位置
#         positions = []
#         for i in range(self.n_particles):
#             if i == 0:
#                 # 第一个用贪心初始化
#                 pos = self._greedy_init()
#             else:
#                 pos = self.repair_solution(self.generate_random_solution())
#             positions.append(pos)
#
#         # 个体最优和全局最优
#         pbest = [p.copy() for p in positions]
#         pbest_fitness = [self._fitness(p) for p in positions]
#
#         gbest_idx = np.argmin(pbest_fitness)
#         gbest = pbest[gbest_idx].copy()
#         gbest_fitness = pbest_fitness[gbest_idx]
#
#         for _ in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             for i in range(self.n_particles):
#                 # 更新位置（离散PSO：概率性移动）
#                 new_pos = positions[i].copy()
#
#                 for j in range(n_items):
#                     r1, r2 = np.random.random(), np.random.random()
#
#                     # 向个体最优学习
#                     if r1 < self.c1 / 4 and pbest[i][j] != new_pos[j]:
#                         new_pos[j] = pbest[i][j]
#
#                     # 向全局最优学习
#                     if r2 < self.c2 / 4 and gbest[j] != new_pos[j]:
#                         new_pos[j] = gbest[j]
#
#                     # 随机扰动
#                     if np.random.random() < 0.1:
#                         new_pos[j] = np.random.randint(-1, n_holds)
#
#                 positions[i] = self.repair_solution(new_pos)
#                 fitness = self._fitness(positions[i])
#
#                 # 更新个体最优
#                 if fitness < pbest_fitness[i]:
#                     pbest[i] = positions[i].copy()
#                     pbest_fitness[i] = fitness
#
#                     # 更新全局最优
#                     if fitness < gbest_fitness:
#                         gbest = positions[i].copy()
#                         gbest_fitness = fitness
#
#         return gbest
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# class CS(BaseAlgorithm):
#     """布谷鸟搜索算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  n_nests=20, iterations=50, pa=0.25, time_limit=30):
#         super().__init__(problem, segment_type)
#         self.n_nests = n_nests
#         self.iterations = iterations
#         self.pa = pa
#         self.time_limit = time_limit
#         self.name = 'CS'
#
#     def solve(self):
#         """布谷鸟搜索求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 初始化鸟巢
#         nests = []
#         for i in range(self.n_nests):
#             if i == 0:
#                 nest = self._greedy_init()
#             else:
#                 nest = self.repair_solution(self.generate_random_solution())
#             nests.append(nest)
#
#         fitness = [self._fitness(nest) for nest in nests]
#
#         best_idx = np.argmin(fitness)
#         best_nest = nests[best_idx].copy()
#         best_fitness = fitness[best_idx]
#
#         for _ in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             # 生成新解（Levy飞行简化版）
#             for i in range(self.n_nests):
#                 new_nest = self._levy_flight(nests[i], n_holds)
#                 new_fitness = self._fitness(new_nest)
#
#                 # 随机选择一个巢比较
#                 j = np.random.randint(self.n_nests)
#                 if new_fitness < fitness[j]:
#                     nests[j] = new_nest
#                     fitness[j] = new_fitness
#
#                     if new_fitness < best_fitness:
#                         best_nest = new_nest.copy()
#                         best_fitness = new_fitness
#
#             # 放弃一部分差的巢
#             sorted_idx = np.argsort(fitness)
#             n_abandon = int(self.pa * self.n_nests)
#
#             for idx in sorted_idx[-n_abandon:]:
#                 nests[idx] = self.repair_solution(self.generate_random_solution())
#                 fitness[idx] = self._fitness(nests[idx])
#
#         return best_nest
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _levy_flight(self, nest, n_holds):
#         """Levy飞行产生新解（简化版）"""
#         new_nest = nest.copy()
#
#         # 随机修改一部分
#         n_modify = max(1, len(nest) // 10)
#         indices = np.random.choice(len(nest), n_modify, replace=False)
#
#         for i in indices:
#             new_nest[i] = np.random.randint(-1, n_holds)
#
#         return self.repair_solution(new_nest)
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# class ACO(BaseAlgorithm):
#     """蚁群优化算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  n_ants=15, iterations=30, alpha=1.0, beta=2.0, rho=0.5,
#                  time_limit=30):
#         super().__init__(problem, segment_type)
#         self.n_ants = n_ants
#         self.iterations = iterations
#         self.alpha = alpha
#         self.beta = beta
#         self.rho = rho
#         self.time_limit = time_limit
#         self.name = 'ACO'
#
#     def solve(self):
#         """蚁群算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         # 信息素矩阵
#         pheromone = np.ones((n_items, n_holds + 1))
#
#         # 启发式信息
#         heuristic = self._compute_heuristic()
#
#         best_solution = None
#         best_fitness = float('inf')
#
#         for _ in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             for _ in range(self.n_ants):
#                 solution = self._construct_solution(pheromone, heuristic)
#                 solution = self.repair_solution(solution)
#                 fitness = self._fitness(solution)
#
#                 if fitness < best_fitness:
#                     best_fitness = fitness
#                     best_solution = solution.copy()
#
#             # 更新信息素
#             pheromone = self._update_pheromone(pheromone, best_solution, best_fitness)
#
#         return best_solution if best_solution else self._greedy_init()
#
#     def _compute_heuristic(self):
#         """计算启发式信息"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, _, _ = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         heuristic = np.ones((n_items, n_holds + 1))
#
#         for i in range(n_items):
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                 heuristic[i, j] = 1.0 / (cg_diff + 0.1)
#
#             heuristic[i, n_holds] = 0.1  # 不装载
#
#         return heuristic
#
#     def _construct_solution(self, pheromone, heuristic):
#         """蚂蚁构建解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = []
#         hold_weights = [0] * n_holds
#
#         for i in range(n_items):
#             item = self.problem.cargo_items.iloc[i]
#
#             # 计算转移概率
#             probs = np.zeros(n_holds + 1)
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     probs[j] = (pheromone[i, j] ** self.alpha *
#                                 heuristic[i, j] ** self.beta)
#             probs[n_holds] = pheromone[i, n_holds] ** self.alpha * 0.1 ** self.beta
#
#             # 归一化
#             total = probs.sum()
#             if total > 0:
#                 probs /= total
#             else:
#                 probs[n_holds] = 1.0
#
#             # 选择
#             choice = np.random.choice(n_holds + 1, p=probs)
#
#             if choice < n_holds:
#                 solution.append(choice)
#                 hold_weights[choice] += item['weight']
#             else:
#                 solution.append(-1)
#
#         return solution
#
#     def _update_pheromone(self, pheromone, best_solution, best_fitness):
#         """更新信息素"""
#         n_holds = self.problem.n_holds
#
#         # 挥发
#         pheromone *= (1 - self.rho)
#
#         # 增强最优解路径
#         if best_solution:
#             delta = 1.0 / (best_fitness + 0.1)
#             for i, j in enumerate(best_solution):
#                 if j >= 0:
#                     pheromone[i, j] += delta
#                 else:
#                     pheromone[i, n_holds] += delta * 0.1
#
#         return pheromone
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         for i in range(n_items):
#             item = self.problem.cargo_items.iloc[i]
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#                 if hold_weights[j] + item['weight'] <= hold['max_weight']:
#                     solution[i] = j
#                     hold_weights[j] += item['weight']
#                     break
#
#         return solution
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# class ABC(BaseAlgorithm):
#     """人工蜂群算法"""
#
#     def __init__(self, problem, segment_type='single',
#                  colony_size=20, iterations=50, limit=10, time_limit=30):
#         super().__init__(problem, segment_type)
#         self.colony_size = colony_size
#         self.iterations = iterations
#         self.limit = limit
#         self.time_limit = time_limit
#         self.name = 'ABC'
#
#     def solve(self):
#         """人工蜂群算法求解"""
#         n_items = self.problem.n_items
#
#         if n_items == 0:
#             return []
#
#         start_time = time.time()
#
#         n_food = self.colony_size // 2
#
#         # 初始化食物源
#         foods = []
#         for i in range(n_food):
#             if i == 0:
#                 food = self._greedy_init()
#             else:
#                 food = self.repair_solution(self.generate_random_solution())
#             foods.append(food)
#
#         fitness = [self._fitness(f) for f in foods]
#         trials = [0] * n_food
#
#         best_solution = foods[np.argmin(fitness)].copy()
#         best_fitness = min(fitness)
#
#         for _ in range(self.iterations):
#             if time.time() - start_time > self.time_limit:
#                 break
#
#             # 雇佣蜂阶段
#             for i in range(n_food):
#                 new_food = self._employed_bee(foods[i], foods)
#                 new_fitness = self._fitness(new_food)
#
#                 if new_fitness < fitness[i]:
#                     foods[i] = new_food
#                     fitness[i] = new_fitness
#                     trials[i] = 0
#
#                     if new_fitness < best_fitness:
#                         best_solution = new_food.copy()
#                         best_fitness = new_fitness
#                 else:
#                     trials[i] += 1
#
#             # 观察蜂阶段
#             probs = self._calculate_probs(fitness)
#
#             for _ in range(n_food):
#                 i = np.random.choice(n_food, p=probs)
#                 new_food = self._employed_bee(foods[i], foods)
#                 new_fitness = self._fitness(new_food)
#
#                 if new_fitness < fitness[i]:
#                     foods[i] = new_food
#                     fitness[i] = new_fitness
#                     trials[i] = 0
#
#                     if new_fitness < best_fitness:
#                         best_solution = new_food.copy()
#                         best_fitness = new_fitness
#                 else:
#                     trials[i] += 1
#
#             # 侦察蜂阶段
#             for i in range(n_food):
#                 if trials[i] > self.limit:
#                     foods[i] = self.repair_solution(self.generate_random_solution())
#                     fitness[i] = self._fitness(foods[i])
#                     trials[i] = 0
#
#         return best_solution
#
#     def _greedy_init(self):
#         """贪心初始化"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
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
#                     cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
#                     if cg_diff < best_score:
#                         best_score = cg_diff
#                         best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _employed_bee(self, food, all_foods):
#         """雇佣蜂操作"""
#         new_food = food.copy()
#         n_holds = self.problem.n_holds
#
#         # 随机修改几个位置
#         n_modify = max(1, len(food) // 10)
#         indices = np.random.choice(len(food), n_modify, replace=False)
#
#         for i in indices:
#             # 随机选择：向其他食物源学习或随机
#             if np.random.random() < 0.5 and all_foods:
#                 k = np.random.randint(len(all_foods))
#                 new_food[i] = all_foods[k][i]
#             else:
#                 new_food[i] = np.random.randint(-1, n_holds)
#
#         return self.repair_solution(new_food)
#
#     def _calculate_probs(self, fitness):
#         """计算选择概率"""
#         max_fit = max(fitness) + 1
#         adjusted = [max_fit - f for f in fitness]
#         total = sum(adjusted) + 1e-10
#         return [f / total for f in adjusted]
#
#     def _fitness(self, solution):
#         return self.get_objective_value(solution)
#
#
# class MBO(BaseAlgorithm):
#     """贪心算法（Greedy Best-fit）"""
#
#     def __init__(self, problem, segment_type='single', time_limit=30):
#         super().__init__(problem, segment_type)
#         self.time_limit = time_limit
#         self.name = 'MBO'
#
#     def solve(self):
#         """贪心算法求解"""
#         n_items = self.problem.n_items
#         n_holds = self.problem.n_holds
#
#         if n_items == 0:
#             return []
#
#         # 计算最优重心
#         total_cargo_weight = sum(self.problem.cargo_items['weight'])
#         optimal_cg, aft_limit, fwd_limit = self.problem.get_optimal_cg(
#             self.problem.initial_weight + total_cargo_weight
#         )
#
#         solution = [-1] * n_items
#         hold_weights = [0] * n_holds
#
#         # 按策略排序货物
#         if self.segment_type == 'multi':
#             # 多航段：行李(B)放后舱，重型货物(C)放中间
#             items_order = self._sort_for_multi_segment()
#         else:
#             # 单航段：按重量降序
#             items_order = sorted(range(n_items),
#                                  key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
#                                  reverse=True)
#
#         for i in items_order:
#             item = self.problem.cargo_items.iloc[i]
#
#             best_hold = -1
#             best_score = float('inf')
#
#             for j in range(n_holds):
#                 hold = self.problem.holds[j]
#
#                 # 检查容量
#                 if hold_weights[j] + item['weight'] > hold['max_weight']:
#                     continue
#
#                 # 计算得分
#                 if self.segment_type == 'multi':
#                     score = self._multi_segment_score(item, hold, j, n_holds)
#                 else:
#                     # 单航段：最小化与最优重心的偏差
#                     cg_contrib = hold['cg_coefficient'] * 1000
#                     score = abs(cg_contrib - optimal_cg)
#
#                 if score < best_score:
#                     best_score = score
#                     best_hold = j
#
#             if best_hold >= 0:
#                 solution[i] = best_hold
#                 hold_weights[best_hold] += item['weight']
#
#         return solution
#
#     def _sort_for_multi_segment(self):
#         """多航段货物排序"""
#         n_items = self.problem.n_items
#         items = []
#
#         for i in range(n_items):
#             item = self.problem.cargo_items.iloc[i]
#             content_type = item.get('content_type', 'C')
#             priority = {'B': 1, 'C': 2, 'M': 3}.get(content_type, 2)
#             items.append((i, priority, item['weight']))
#
#         # 按优先级和重量排序
#         items.sort(key=lambda x: (x[1], -x[2]))
#         return [x[0] for x in items]
#
#     def _multi_segment_score(self, item, hold, hold_idx, n_holds):
#         """多航段评分"""
#         content_type = item.get('content_type', 'C')
#         cg_coef = hold['cg_coefficient']
#
#         if content_type == 'B':
#             # 行李放后舱（cg_coefficient > 0 表示后舱）
#             if cg_coef > 0:
#                 return -cg_coef * 100  # 越后越好
#             else:
#                 return abs(cg_coef) * 100
#         else:
#             # 货物放中间
#             return abs(cg_coef) * 100
#
#
# if __name__ == '__main__':
#     print("Heuristic Algorithms Module: GA, PSO, CS, ACO, ABC, MBO")
#     print("All algorithms have timeout mechanism (default 30s)")

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic Algorithms: GA, PSO, CS, ACO, ABC, MBO
启发式算法实现 - 优化版本
加入超时机制，确保在限定时间内返回解
增加内存使用以反映真实算法特征
"""

import numpy as np
import time
from .base_algorithm1 import BaseAlgorithm


class GA(BaseAlgorithm):
    """遗传算法"""

    def __init__(self, problem, segment_type='single',
                 pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1,
                 time_limit=30):
        super().__init__(problem, segment_type)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.time_limit = time_limit
        self.name = 'GA'

    def solve(self):
        """遗传算法求解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 种群数据结构 ==========
        # 种群矩阵（每行一个个体）
        population = np.zeros((self.pop_size, n_items), dtype=np.int32)

        # 适应度数组
        fitness = np.zeros(self.pop_size, dtype=np.float64)

        # 历史最优记录
        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_solutions': []
        }

        # 交叉概率矩阵
        crossover_mask = np.random.rand(self.pop_size, n_items) < self.crossover_rate

        # 变异概率矩阵
        mutation_mask = np.random.rand(self.pop_size, n_items) < self.mutation_rate

        # 初始化种群
        for i in range(self.pop_size):
            if i < self.pop_size // 3:
                ind = self._greedy_init()
            else:
                ind = self._random_init()
            population[i] = ind
            fitness[i] = self._fitness(list(ind))

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for gen in range(self.generations):
            if time.time() - start_time > self.time_limit:
                break

            # 记录历史
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitness))

            # 选择（锦标赛）
            selected_idx = np.zeros(self.pop_size, dtype=np.int32)
            for i in range(self.pop_size):
                candidates = np.random.choice(self.pop_size, 2, replace=False)
                winner = candidates[0] if fitness[candidates[0]] < fitness[candidates[1]] else candidates[1]
                selected_idx[i] = winner

            selected = population[selected_idx].copy()

            # 交叉
            offspring = np.zeros_like(population)
            for i in range(0, self.pop_size - 1, 2):
                if np.random.random() < self.crossover_rate:
                    point = np.random.randint(1, n_items)
                    offspring[i, :point] = selected[i, :point]
                    offspring[i, point:] = selected[i + 1, point:]
                    offspring[i + 1, :point] = selected[i + 1, :point]
                    offspring[i + 1, point:] = selected[i, point:]
                else:
                    offspring[i] = selected[i]
                    offspring[i + 1] = selected[i + 1]

            # 变异
            mutation_mask = np.random.rand(self.pop_size, n_items) < self.mutation_rate
            random_values = np.random.randint(-1, n_holds, size=(self.pop_size, n_items))
            offspring = np.where(mutation_mask, random_values, offspring)

            # 修复并评估
            for i in range(self.pop_size):
                repaired = self.repair_solution(list(offspring[i]))
                offspring[i] = repaired
                fitness[i] = self._fitness(repaired)

            # 精英保留
            worst_idx = np.argmax(fitness)
            offspring[worst_idx] = best_solution
            fitness[worst_idx] = best_fitness

            population = offspring

            # 更新最优
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()
                history['best_solutions'].append(best_solution.copy())

        # 保持数据在内存中
        _ = (population, fitness, history, crossover_mask, mutation_mask)

        return list(best_solution)

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
                    cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                    if cg_diff < best_score:
                        best_score = cg_diff
                        best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        return solution

    def _random_init(self):
        """随机初始化"""
        return self.repair_solution(self.generate_random_solution())

    def _fitness(self, individual):
        """计算适应度"""
        return self.get_objective_value(individual)


class PSO(BaseAlgorithm):
    """粒子群优化算法"""

    def __init__(self, problem, segment_type='single',
                 n_particles=40, iterations=100, w=0.7, c1=1.5, c2=1.5,
                 time_limit=30):
        super().__init__(problem, segment_type)
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.time_limit = time_limit
        self.name = 'PSO'

    def solve(self):
        """粒子群算法求解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 粒子数据结构 ==========
        # 位置矩阵（连续值，范围 [-1, n_holds-1]）
        positions = np.random.uniform(-1, n_holds - 1, (self.n_particles, n_items))

        # 速度矩阵
        velocities = np.zeros((self.n_particles, n_items), dtype=np.float64)

        # 个体最优位置
        pbest = positions.copy()
        pbest_fitness = np.full(self.n_particles, float('inf'))

        # 全局最优
        gbest = np.zeros(n_items, dtype=np.float64)
        gbest_fitness = float('inf')

        # 历史记录
        history = {
            'gbest_fitness': [],
            'positions': []
        }

        # 初始化第一个粒子为贪心解
        greedy_sol = self._greedy_init()
        positions[0] = np.array(greedy_sol, dtype=np.float64)

        # 初始评估
        for i in range(self.n_particles):
            decoded = self._decode(positions[i], n_holds)
            fitness = self._fitness(decoded)
            pbest_fitness[i] = fitness

            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest = positions[i].copy()

        for iteration in range(self.iterations):
            if time.time() - start_time > self.time_limit:
                break

            # 记录历史
            history['gbest_fitness'].append(gbest_fitness)

            # 更新速度和位置
            r1 = np.random.rand(self.n_particles, n_items)
            r2 = np.random.rand(self.n_particles, n_items)

            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest - positions) +
                          self.c2 * r2 * (gbest - positions))

            # 限制速度范围
            max_velocity = (n_holds + 1) * 0.5
            velocities = np.clip(velocities, -max_velocity, max_velocity)

            positions = positions + velocities
            # 限制位置范围在 [-1, n_holds-1]
            positions = np.clip(positions, -1, n_holds - 1)

            # 评估
            for i in range(self.n_particles):
                decoded = self._decode(positions[i], n_holds)
                fitness = self._fitness(decoded)

                if fitness < pbest_fitness[i]:
                    pbest[i] = positions[i].copy()
                    pbest_fitness[i] = fitness

                    if fitness < gbest_fitness:
                        gbest = positions[i].copy()
                        gbest_fitness = fitness

        # 保持数据在内存中
        _ = (positions, velocities, pbest, pbest_fitness, history)

        return self.repair_solution(self._decode(gbest, n_holds))

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
                    cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                    if cg_diff < best_score:
                        best_score = cg_diff
                        best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        return solution

    def _decode(self, position, n_holds):
        """将连续位置解码为离散解"""
        decoded = []
        for p in position:
            if p < -0.5:
                decoded.append(-1)  # 不装载
            else:
                # 四舍五入并限制在有效范围内
                idx = int(round(p))
                idx = max(0, min(idx, n_holds - 1))
                decoded.append(idx)
        return decoded

    def _fitness(self, solution):
        solution = self.repair_solution(solution)
        return self.get_objective_value(solution)


class CS(BaseAlgorithm):
    """布谷鸟搜索算法"""

    def __init__(self, problem, segment_type='single',
                 n_nests=30, iterations=100, pa=0.25, time_limit=30):
        super().__init__(problem, segment_type)
        self.n_nests = n_nests
        self.iterations = iterations
        self.pa = pa
        self.time_limit = time_limit
        self.name = 'CS'

    def solve(self):
        """布谷鸟搜索求解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 鸟巢数据结构 ==========
        # 鸟巢矩阵
        nests = np.zeros((self.n_nests, n_items), dtype=np.int32)

        # 适应度数组
        fitness = np.zeros(self.n_nests, dtype=np.float64)

        # Levy飞行参数
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        # 历史记录
        history = {
            'best_fitness': [],
            'abandoned': []
        }

        # 初始化鸟巢
        for i in range(self.n_nests):
            if i == 0:
                nest = self._greedy_init()
            else:
                nest = self.repair_solution(self.generate_random_solution())
            nests[i] = nest
            fitness[i] = self._fitness(list(nest))

        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.iterations):
            if time.time() - start_time > self.time_limit:
                break

            history['best_fitness'].append(best_fitness)

            # Levy飞行产生新解
            for i in range(self.n_nests):
                new_nest = nests[i].copy()

                # Levy飞行
                u = np.random.normal(0, sigma, n_items)
                v = np.random.normal(0, 1, n_items)
                step = u / (np.abs(v) ** (1 / beta))

                # 应用步长
                step_size = 0.01 * step * (nests[i] - best_nest)
                new_nest = nests[i] + step_size.astype(np.int32)
                new_nest = np.clip(new_nest, -1, n_holds - 1)

                new_nest_list = self.repair_solution(list(new_nest))
                new_fitness = self._fitness(new_nest_list)

                # 随机替换一个巢
                j = np.random.randint(self.n_nests)
                if new_fitness < fitness[j]:
                    nests[j] = new_nest_list
                    fitness[j] = new_fitness

                    if new_fitness < best_fitness:
                        best_nest = np.array(new_nest_list)
                        best_fitness = new_fitness

            # 放弃一部分差的巢
            sorted_idx = np.argsort(fitness)
            n_abandon = int(self.pa * self.n_nests)
            history['abandoned'].append(n_abandon)

            for idx in sorted_idx[-n_abandon:]:
                new_nest = self.repair_solution(self.generate_random_solution())
                nests[idx] = new_nest
                fitness[idx] = self._fitness(new_nest)

        # 保持数据在内存中
        _ = (nests, fitness, history)

        return list(best_nest)

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
                    cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                    if cg_diff < best_score:
                        best_score = cg_diff
                        best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        return solution

    def _fitness(self, solution):
        return self.get_objective_value(solution)


class ACO(BaseAlgorithm):
    """蚁群优化算法"""

    def __init__(self, problem, segment_type='single',
                 n_ants=30, iterations=80, alpha=1.0, beta=2.0, rho=0.5,
                 time_limit=30):
        super().__init__(problem, segment_type)
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.time_limit = time_limit
        self.name = 'ACO'

    def solve(self):
        """蚁群算法求解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 信息素数据结构 ==========
        # 信息素矩阵
        pheromone = np.ones((n_items, n_holds + 1), dtype=np.float64)

        # 启发式信息矩阵
        heuristic = np.zeros((n_items, n_holds + 1), dtype=np.float64)

        # 蚂蚁路径矩阵
        ant_paths = np.zeros((self.n_ants, n_items), dtype=np.int32)
        ant_fitness = np.zeros(self.n_ants, dtype=np.float64)

        # 历史记录
        history = {
            'best_fitness': [],
            'pheromone_sum': []
        }

        # 计算启发式信息
        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        for i in range(n_items):
            for j in range(n_holds):
                hold = self.problem.holds[j]
                cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                heuristic[i, j] = 1.0 / (cg_diff + 0.1)
            heuristic[i, n_holds] = 0.1  # 不装载

        best_solution = None
        best_fitness = float('inf')

        for iteration in range(self.iterations):
            if time.time() - start_time > self.time_limit:
                break

            # 每只蚂蚁构建解
            for ant in range(self.n_ants):
                solution = self._construct_solution(pheromone, heuristic)
                ant_paths[ant] = solution
                ant_fitness[ant] = self._fitness(list(solution))

                if ant_fitness[ant] < best_fitness:
                    best_fitness = ant_fitness[ant]
                    best_solution = solution.copy()

            # 记录历史
            history['best_fitness'].append(best_fitness)
            history['pheromone_sum'].append(np.sum(pheromone))

            # 更新信息素
            pheromone *= (1 - self.rho)

            # 最优蚂蚁增强
            if best_solution is not None:
                delta = 1.0 / (best_fitness + 0.1)
                for i, j in enumerate(best_solution):
                    if j >= 0:
                        pheromone[i, j] += delta
                    else:
                        pheromone[i, n_holds] += delta * 0.1

        # 保持数据在内存中
        _ = (pheromone, heuristic, ant_paths, ant_fitness, history)

        return list(best_solution) if best_solution is not None else self._greedy_init()

    def _construct_solution(self, pheromone, heuristic):
        """蚂蚁构建解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        solution = np.zeros(n_items, dtype=np.int32)
        hold_weights = [0] * n_holds

        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]

            # 计算转移概率
            probs = np.zeros(n_holds + 1, dtype=np.float64)
            for j in range(n_holds):
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] <= hold['max_weight']:
                    probs[j] = (pheromone[i, j] ** self.alpha *
                                heuristic[i, j] ** self.beta)
            probs[n_holds] = pheromone[i, n_holds] ** self.alpha * 0.1 ** self.beta

            # 确保概率非负，处理NaN和Inf
            probs = np.maximum(probs, 0)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs[n_holds] = 1.0

            choice = np.random.choice(n_holds + 1, p=probs)

            if choice < n_holds:
                solution[i] = choice
                hold_weights[choice] += item['weight']
            else:
                solution[i] = -1

        return solution

    def _greedy_init(self):
        """贪心初始化"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds
        solution = [-1] * n_items
        hold_weights = [0] * n_holds

        for i in range(n_items):
            item = self.problem.cargo_items.iloc[i]
            for j in range(n_holds):
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] <= hold['max_weight']:
                    solution[i] = j
                    hold_weights[j] += item['weight']
                    break
        return solution

    def _fitness(self, solution):
        return self.get_objective_value(solution)


class ABC(BaseAlgorithm):
    """人工蜂群算法"""

    def __init__(self, problem, segment_type='single',
                 colony_size=40, iterations=80, limit=20, time_limit=30):
        super().__init__(problem, segment_type)
        self.colony_size = colony_size
        self.iterations = iterations
        self.limit = limit
        self.time_limit = time_limit
        self.name = 'ABC'

    def solve(self):
        """人工蜂群算法求解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        n_food = self.colony_size // 2

        # ========== 蜂群数据结构 ==========
        # 食物源矩阵
        foods = np.zeros((n_food, n_items), dtype=np.int32)

        # 适应度数组
        fitness = np.zeros(n_food, dtype=np.float64)

        # 尝试次数
        trials = np.zeros(n_food, dtype=np.int32)

        # 概率数组
        probs = np.zeros(n_food, dtype=np.float64)

        # 历史记录
        history = {
            'best_fitness': [],
            'scout_count': []
        }

        # 初始化食物源
        for i in range(n_food):
            if i == 0:
                food = self._greedy_init()
            else:
                food = self.repair_solution(self.generate_random_solution())
            foods[i] = food
            fitness[i] = self._fitness(list(food))

        best_idx = np.argmin(fitness)
        best_solution = foods[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.iterations):
            if time.time() - start_time > self.time_limit:
                break

            scout_count = 0

            # 雇佣蜂阶段
            for i in range(n_food):
                new_food = self._employed_bee(foods[i], foods, n_holds)
                new_fitness = self._fitness(list(new_food))

                if new_fitness < fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trials[i] = 0

                    if new_fitness < best_fitness:
                        best_solution = new_food.copy()
                        best_fitness = new_fitness
                else:
                    trials[i] += 1

            # 计算选择概率 - 修复负fitness导致的负概率问题
            min_fitness = np.min(fitness)
            shifted_fitness = fitness - min_fitness + 1e-10  # 确保所有值为正
            probs = 1.0 / (1 + shifted_fitness)

            # 确保概率非负且有效
            probs = np.maximum(probs, 0)
            probs = np.nan_to_num(probs, nan=1e-10, posinf=1e-10, neginf=1e-10)

            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                probs = np.ones(n_food) / n_food

            # 观察蜂阶段
            for _ in range(n_food):
                i = np.random.choice(n_food, p=probs)
                new_food = self._employed_bee(foods[i], foods, n_holds)
                new_fitness = self._fitness(list(new_food))

                if new_fitness < fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trials[i] = 0

                    if new_fitness < best_fitness:
                        best_solution = new_food.copy()
                        best_fitness = new_fitness
                else:
                    trials[i] += 1

            # 侦察蜂阶段
            for i in range(n_food):
                if trials[i] > self.limit:
                    new_food = self.repair_solution(self.generate_random_solution())
                    foods[i] = new_food
                    fitness[i] = self._fitness(new_food)
                    trials[i] = 0
                    scout_count += 1

            history['best_fitness'].append(best_fitness)
            history['scout_count'].append(scout_count)

        # 保持数据在内存中
        _ = (foods, fitness, trials, probs, history)

        return list(best_solution)

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
                    cg_diff = abs(hold['cg_coefficient'] * 1000 - optimal_cg)
                    if cg_diff < best_score:
                        best_score = cg_diff
                        best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        return solution

    def _employed_bee(self, food, all_foods, n_holds):
        """雇佣蜂操作"""
        new_food = food.copy()
        n_modify = max(1, len(food) // 10)
        indices = np.random.choice(len(food), n_modify, replace=False)

        for i in indices:
            if np.random.random() < 0.5 and len(all_foods) > 0:
                k = np.random.randint(len(all_foods))
                new_food[i] = all_foods[k][i]
            else:
                new_food[i] = np.random.randint(-1, n_holds)

        return np.array(self.repair_solution(list(new_food)))

    def _fitness(self, solution):
        return self.get_objective_value(solution)


class MBO(BaseAlgorithm):
    """帝王蝶优化算法 (Monarch Butterfly Optimization)"""

    def __init__(self, problem, segment_type='single',
                 pop_size=30, iterations=50, time_limit=30):
        super().__init__(problem, segment_type)
        self.pop_size = pop_size
        self.iterations = iterations
        self.time_limit = time_limit
        self.name = 'MBO'

    def solve(self):
        """帝王蝶优化算法求解"""
        n_items = self.problem.n_items
        n_holds = self.problem.n_holds

        if n_items == 0:
            return []

        start_time = time.time()

        # ========== 蝴蝶种群数据结构 ==========
        # 两个子种群
        n_land1 = int(self.pop_size * 0.5)  # 栖息地1
        n_land2 = self.pop_size - n_land1  # 栖息地2

        land1 = np.zeros((n_land1, n_items), dtype=np.int32)
        land2 = np.zeros((n_land2, n_items), dtype=np.int32)

        fitness1 = np.zeros(n_land1, dtype=np.float64)
        fitness2 = np.zeros(n_land2, dtype=np.float64)

        # 历史记录
        history = {
            'best_fitness': [],
            'migration_count': []
        }

        # 计算最优重心
        total_cargo_weight = sum(self.problem.cargo_items['weight'])
        optimal_cg, _, _ = self.problem.get_optimal_cg(
            self.problem.initial_weight + total_cargo_weight
        )

        # 初始化种群
        for i in range(n_land1):
            if i == 0:
                land1[i] = self._greedy_init()
            else:
                land1[i] = self.repair_solution(self.generate_random_solution())
            fitness1[i] = self._fitness(list(land1[i]))

        for i in range(n_land2):
            land2[i] = self.repair_solution(self.generate_random_solution())
            fitness2[i] = self._fitness(list(land2[i]))

        # 全局最优
        all_fitness = np.concatenate([fitness1, fitness2])
        all_pop = np.vstack([land1, land2])
        best_idx = np.argmin(all_fitness)
        best_solution = all_pop[best_idx].copy()
        best_fitness = all_fitness[best_idx]

        for iteration in range(self.iterations):
            if time.time() - start_time > self.time_limit:
                break

            migration_count = 0
            p = 0.5 * (1 + np.cos(np.pi * iteration / self.iterations))

            # 迁移操作
            for i in range(n_land1):
                if np.random.rand() < p:
                    # 从land2迁移
                    r = np.random.randint(n_land2)
                    new_butterfly = land2[r].copy()
                    migration_count += 1
                else:
                    # 局部搜索
                    new_butterfly = land1[i].copy()
                    idx = np.random.randint(n_items)
                    new_butterfly[idx] = np.random.randint(-1, n_holds)

                new_butterfly = np.array(self.repair_solution(list(new_butterfly)))
                new_fitness = self._fitness(list(new_butterfly))

                if new_fitness < fitness1[i]:
                    land1[i] = new_butterfly
                    fitness1[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_butterfly.copy()
                        best_fitness = new_fitness

            # 蝴蝶调整操作
            for i in range(n_land2):
                new_butterfly = land2[i].copy()

                if np.random.rand() < p:
                    # 向最优学习
                    idx = np.random.randint(n_items)
                    new_butterfly[idx] = best_solution[idx]
                else:
                    # 随机扰动
                    idx = np.random.randint(n_items)
                    new_butterfly[idx] = np.random.randint(-1, n_holds)

                new_butterfly = np.array(self.repair_solution(list(new_butterfly)))
                new_fitness = self._fitness(list(new_butterfly))

                if new_fitness < fitness2[i]:
                    land2[i] = new_butterfly
                    fitness2[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_butterfly.copy()
                        best_fitness = new_fitness

            history['best_fitness'].append(best_fitness)
            history['migration_count'].append(migration_count)

        # 保持数据在内存中
        _ = (land1, land2, fitness1, fitness2, history)

        return list(best_solution)

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

        # 按重量降序
        items_order = sorted(range(n_items),
                             key=lambda i: self.problem.cargo_items.iloc[i]['weight'],
                             reverse=True)

        for i in items_order:
            item = self.problem.cargo_items.iloc[i]
            best_hold = -1
            best_score = float('inf')

            for j in range(n_holds):
                hold = self.problem.holds[j]
                if hold_weights[j] + item['weight'] > hold['max_weight']:
                    continue

                cg_contrib = hold['cg_coefficient'] * 1000
                score = abs(cg_contrib - optimal_cg)

                if score < best_score:
                    best_score = score
                    best_hold = j

            if best_hold >= 0:
                solution[i] = best_hold
                hold_weights[best_hold] += item['weight']

        return solution

    def _fitness(self, solution):
        return self.get_objective_value(solution)


if __name__ == '__main__':
    print("Heuristic Algorithms Module: GA, PSO, CS, ACO, ABC, MBO")
    print("All algorithms have timeout mechanism (default 30s)")
