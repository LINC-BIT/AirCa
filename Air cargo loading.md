# Air Cargo Loading

This task aims to study the optimization process of cargo loading to improve fuel efficiency while meeting the constraint of the aircraft center of gravity.


# Required Python Packages

To install all required packages, run the following pip commands:
```bash
pip install numpy
pip install scipy
pip install pyswarms
pip install deap
pip install pulp
```

## Example 1: COM
Combinatorial Optimization Model solves discrete optimization tasks by searching for an optimal arrangement among a finite set of feasible solutions.
Run the following command in Python:
```bash
python COM.py
```

## Example 2: IOM
Improved Combinatorial Optimization Model obtains better solutions for discrete optimization tasks by refining search strategies to more effectively explore feasible configurations.
Run the following command in Python:
```bash
python IOM.py
```

## Example 3: NL-CPLEX
NL-CPLEX addresses non-linear optimization tasks by leveraging branch-and-bound and cutting-plane techniques to efficiently explore the solution space.
Run the following command in Python:
```bash
python NL_CPLEX.py
```

## Example 4: SDCCLPM
Stochastic-Demand Cargo Container Loading Plan Model optimizes container loading configurations under demand uncertainty by incorporating probabilistic approaches to balance capacity and cost requirements.
Run the following command in Python:
```bash
python SDCCLPM.py
```

## Example 5: MLIP
Mixed Integer Linear Program finds optimal solutions to discrete optimization problems by combining integer constraints with linear relationships in a branch-and-bound search process.
Run the following command in Python:
```bash
python MLIP.py
```

## Example 6: MLIP-WBP
MLIP-WBP optimizes weighted bin packing by employing a Mixed Integer Linear Programming formulation to balance item distribution and capacity constraints.
Run the following command in Python:
```bash
python MLIP_WBP.py
```

## Example 7: MLIP-ACLPDD
MLIP-ACLPDD solves advanced cargo loading planning under uncertain demand by incorporating robust constraints into a Mixed Integer Linear Programming framework.
Run the following command in Python:
```bash
python MLIP_ACLPDD.py
```

## Example 8: HGA
Hybrid Genetic Algorithm enhances solution quality by combining evolutionary operators with complementary search techniques to accelerate convergence and explore the solution space more thoroughly.
Run the following command in Python:
```bash
python HGA.py
```

## Example 9: GA-normal
GA-normal employs foundational genetic algorithm operations—selection, crossover, and mutation—to explore solutions within a population-based search framework.
Run the following command in Python:
```bash
python GA_normal.py
```

## Example 10: DMOPSO
Discrete Multi-Objective Particle Swarm Optimization locates Pareto-optimal solutions in discrete search spaces by adapting swarm-based velocity and position update mechanisms to address multiple conflicting objectives.
Run the following command in Python:
```bash
python DMOPSO.py
```

## Example 11: PSO-normal
PSO-normal employs the basic velocity and position update rules, guided by personal and global best solutions, to iteratively converge on an optimal search space configuration.
Run the following command in Python:
```bash
python PSO_normal.py
```

## Example 12: RCH
Randomized Constructive Heuristic incrementally constructs feasible solutions by integrating stochastic choices during each step, thus diversifying the search process and enhancing solution discovery.
Run the following command in Python:
```bash
python RCH.py
```



