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
![image](https://github.com/user-attachments/assets/1655d1ea-7c69-4a43-a102-978cb3b9740a)

## Example 2: IOM
Improved Combinatorial Optimization Model obtains better solutions for discrete optimization tasks by refining search strategies to more effectively explore feasible configurations.
Run the following command in Python:
```bash
python IOM.py
```
![image](https://github.com/user-attachments/assets/1d99a4a6-4a11-42e0-94b2-ab80d67f7c11)

## Example 3: NL-CPLEX
NL-CPLEX addresses non-linear optimization tasks by leveraging branch-and-bound and cutting-plane techniques to efficiently explore the solution space.
Run the following command in Python:
```bash
python NL_CPLEX.py
```
![image](https://github.com/user-attachments/assets/7ba87558-5303-4763-9a3d-774c20d96957)

## Example 4: SDCCLPM
Stochastic-Demand Cargo Container Loading Plan Model optimizes container loading configurations under demand uncertainty by incorporating probabilistic approaches to balance capacity and cost requirements.
Run the following command in Python:
```bash
python SDCCLPM.py
```
![image](https://github.com/user-attachments/assets/2db275a6-0ae0-4e9b-9674-f39d40ddba73)


## Example 5: MLIP
Mixed Integer Linear Program finds optimal solutions to discrete optimization problems by combining integer constraints with linear relationships in a branch-and-bound search process.
Run the following command in Python:
```bash
python MLIP.py
```
![image](https://github.com/user-attachments/assets/4e3e2ac1-b8c7-4f98-8172-83a172ecc62d)

## Example 6: MLIP-WBP
MLIP-WBP optimizes weighted bin packing by employing a Mixed Integer Linear Programming formulation to balance item distribution and capacity constraints.
Run the following command in Python:
```bash
python MLIP_WBP.py
```
![image](https://github.com/user-attachments/assets/9dfa9f74-a295-4e6c-9caf-62225aa0e493)

## Example 7: MLIP-ACLPDD
MLIP-ACLPDD solves advanced cargo loading planning under uncertain demand by incorporating robust constraints into a Mixed Integer Linear Programming framework.
Run the following command in Python:
```bash
python MLIP_ACLPDD.py
```
![image](https://github.com/user-attachments/assets/2eb65571-9cc1-4ea2-9897-d810d171b9d6)

## Example 8: HGA
Hybrid Genetic Algorithm enhances solution quality by combining evolutionary operators with complementary search techniques to accelerate convergence and explore the solution space more thoroughly.
Run the following command in Python:
```bash
python HGA.py
```
![image](https://github.com/user-attachments/assets/ec96c8ed-b4b0-4073-afe1-6343e5b98680)

## Example 9: GA-normal
GA-normal employs foundational genetic algorithm operations—selection, crossover, and mutation—to explore solutions within a population-based search framework.
Run the following command in Python:
```bash
python GA_normal.py
```
![image](https://github.com/user-attachments/assets/8c712fcd-a58f-403d-b5e4-8e3fd568ba1d)

## Example 10: DMOPSO
Discrete Multi-Objective Particle Swarm Optimization locates Pareto-optimal solutions in discrete search spaces by adapting swarm-based velocity and position update mechanisms to address multiple conflicting objectives.
Run the following command in Python:
```bash
python DMOPSO.py
```
![image](https://github.com/user-attachments/assets/70826025-6d4c-4081-a104-e4dff934928b)

## Example 11: PSO-normal
PSO-normal employs the basic velocity and position update rules, guided by personal and global best solutions, to iteratively converge on an optimal search space configuration.
Run the following command in Python:
```bash
python PSO_normal.py
```
![image](https://github.com/user-attachments/assets/dac97ef1-afa0-4f74-9c26-783fe5a06b53)

## Example 12: RCH
Randomized Constructive Heuristic incrementally constructs feasible solutions by integrating stochastic choices during each step, thus diversifying the search process and enhancing solution discovery.
Run the following command in Python:
```bash
python RCH.py
```
****![image](https://github.com/user-attachments/assets/db2a43b4-b0fd-4aea-bbde-63d94fbca6bd)



