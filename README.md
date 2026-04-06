
<a href="https://linc-bit.github.io/AirCa/html-page/index.html">
<img width="1415" height="385" alt="1" src="https://github.com/user-attachments/assets/1dde8fd2-a52c-445d-91bb-dc4a3cafa112" />
</a>


<div align="center">
<a href="https://linc-bit.github.io/AirCa/html-page/index.html#quickstart" style="text-decoration:none; outline:none;">
    <img src="quick start.png" alt="Quick Start" height="38" style="border:0;">
</a>
  &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/LINC-BIT/AirCa/tree/main/code" style="text-decoration:none; outline:none;">
    <img src="view source code.png" alt="View Source Code" height="38" style="border:0;">
</a>
</div>


<div align="center">

📌 [Demo](https://linc-bit.github.io/AirCa/html-page/index.html#examples) &nbsp;&nbsp;|&nbsp;&nbsp; 📖 [Overview](https://linc-bit.github.io/AirCa/html-page/index.html#about) &nbsp;&nbsp;|&nbsp;&nbsp; 🤗 [Download](https://huggingface.co/datasets/LINC-BIT/AirCa)

</div>


# AirCa Demo
Users can select from 6 preset experiments, choose an optimization algorithm, and generate a loading plan with one click. Results are visualized directly on a 3D aircraft hold layout (Gif below).


<div align="center">
 
![alt text](<AirCa demo-1.gif>)

</div>

Click here to access AirCa demo: https://linc-bit.github.io/AirCa/html-page/index.html#examples




# Tutorial index

- [1. AirCa](#1-airca)
- [2. Download](#2-download)
- [3. Description](#3-description)
  * [3.1 AirCa data field](#31-airca-data-field)
  * [3.2 Constraints description](#32-constraints-description)
- [4. Tutorials of workloads](#4-tutorials-of-workloads)
  * [4.1 Multi-constraint cargo loading](#41-multi-constraint-cargo-loading)
  * [4.2 Cargo loading with massive variables](#42-cargo-loading-with-massive-variables)
  * [4.3 Multi-segment cargo loading](#43-multi-segment-cargo-loading)
- [5. The AirCa APIs](#5-the-airca-apis)

- [6. References](#6-references)



Dataset Download: https://huggingface.co/datasets/LINC-BIT/AirCa       
Dataset Website: https://linc-bit.github.io/AirCa/html-page/index.html   
Code Link: https://github.com/LINC-BIT/AirCa     
Paper Link:  


# 1. AirCa

 **AirCa** is a publicly available  aircraft cargo loading dataset with  millions of instances from industry. It has three unique characteristics: 
(1) Large-scale, AirCa contains in total 6,071k records and 1,092k flights, covering 6 aircraft types and 425 airports over a total span of 9 months. 
(2) Comprehensive information, AirCa is delivered to provide rich information pertaining to aircraft cargo loading, including detailed cargo characteristic information, loading-event logs, flight destination, and comprehensive loading constraints in practical scenarios.
(3) Diversity,  AirCa aims to increase data diversity from three perspectives: destination diversity, Flight diversity, and Constraint diversity.

![](introduction.jpg)

This figure illustrates an air cargo loading scenario comprising three parts: (1) *Air cargo* has two types: bulk cargo, which consists of individual sub-cargoes, and Unit Load Devices (ULDs), which are pre-packed standardized containers. (2) *Cargo holds* accommodate bulk cargoes (e.g., narrow-body aircraft such as the A320) and ULDs (e.g., wide-body aircraft such as the B777). Notably, when loaded into bulk cargo holds, different cargo detaching granularities produce a massive number of loading options.
(3) *Constraints* determine the feasibility of loading operations, including cargo constraints, cargo hold constraints, and loading operation constraints. In this figure's example, the flight has two destinations, and hence the ULDs are categorized into two segments. Such multi-segment cargo loading further complicates the combinatorial optimization problem.

# 2. Download

AirCa can be used for research purposes. Before you download the dataset, please read these terms.
Then put the data into "./data/raw/".  
The structure of "./data/raw/" should be like:  
```
* ./data/raw/  
    * split_by_aircraft_type    
        * A320.csv   
        * ...    
    * split_by_date  
        * BAKFLGITH_LOADDATA2024-10-12.csv  
        * ...
```
```python
import pandas as pd
>>> import pandas as pd
>>> df = pd.read_csv("BAKFLGITH_LOADDATA2024-10-12.csv")
>>> df.head(3)
       FLIGHT  TYPE DEST  WEIGHT  ... CONT PRIORITY VOLUME  SPECIAL CARGO
0  3744617311  A320  SIN     177  ...  NaN        1    0.0            NaN
1  3744617311  A320  SIN     177  ...  NaN        1    0.0            NaN
2  3744617332  A320  SIN     560  ...  NaN        1    0.0            NaN
```

# 3. Description
## 3.1 AirCa data field
| Data field | Description | Unit/format |
|---|---|---|
| **Cargo information** |  |  |
| Loading order | Record of the cargo loading order | String |
| ID | Unique identifier for cargo | ID |
| Weight | Weight of cargo | String |
| ULD type | Types of cargo including ULD and bulk cargo | String |
| Priority | Cargo loading priority | String |
| Length | Length of cargo | Float |
| Width | Width of cargo | Float |
| Height | Height of cargo | Float |
| Detaching information | The record of detached cargoes | String |
| **Flight information** |  |  |
| Loading time | Record of the cargo loading time | Time |
| Flight ID (anonymity) | Record of the different flights | ID |
| Destination airport | Record of the airport's name | String |
| Segment | The record of whether it is a multi-segment flight | Bool |
| **Aircraft information** |  |  |
| Aircraft type | The type of the aircraft | String |
| Constraints | The constraints of air cargo loading | Constraint format |


## 3.2 Constraints description
| Constraint | Description |
|---|---|
| **Cargo constraints** |  |
| Cargo weight constraint | Maximum weight limit per cargo hold |
| Cargo volume constraint | Maximum volume limit per cargo hold |
| Center of gravity constraint | Forward and backward CG limits |
| **Cargo hold constraints** |  |
| ULD type compatibility | ULD type must match cargo hold specification |
| Cargo hold exclusivity | Each cargo hold accommodates at most one ULD |
| Cargo type validity | Cargo type must be valid for the cargo hold |
| Front/Rear compartment | Short-haul cargo is prioritized for loading in rear cargo hold |
| Special constraint | Certain holds prohibit specific dangerous cargoes |
| ULD correspond constraint | Each cargo hold accepts only compatible ULD types |
| **Loading operation constraints** |  |
| Dangerous cargoes isolation | Minimum distance between incompatible dangerous cargoes |
| Joint weight constraint | Maximum weight limit for adjacent cargo hold combinations |
| Continuous loading | No gaps between cargoes during loading process |
| Loading order constraint | Short-haul cargo positioned outboard of long-haul cargo |

# 4. Tutorials of workloads
We extract three representative workloads from AirCa including multi-constraint cargo loading, cargo loading 
with massive variables, and multi-segment cargo loading. 

| Workload name | Workload characteristic | Challenge |
|---|---|---|
| Multi-constraint cargo loading | High modeling complexity arises from intertwined constraints and heterogeneous cargo/ULD data that require distinct variable sets. | Oversimplified modeling can reduce solution accuracy when algorithms face complex constraint interactions. |
| Cargo loading with massive variables | Cargo detaching at finer granularities dramatically increases decision variables and expands the search space exponentially. | Massive variables slow heuristic optimization because far more candidate solutions must be explored as the search space grows. |
| Multi-segment cargo loading | Multi-stage sequential decisions with dual objectives (minimizing CG offset while maximizing profit) increase problem complexity. | Conflicting objectives require re-balancing across stages, which can increase computation time to achieve feasible and high-quality solutions. |

Before running the workloads, please install the required Python libraries and preprocess the data from the huggingface:

```bash
pip install -r requirements.txt

python data_preprocess.py --input-path /data/raw/ --output-path /data/processed/
```

### Evaluation metrics
The experimental results in Section 4 mainly report CG-related gap, and algorithm runtime.

**CG (center of gravity):** In the weight-and-balance setting, the aircraft CG is represented by the balance arm, denoted by $BA$, which is the horizontal distance from the reference datum to the location of the aircraft center of gravity. In the index-based formulation used by practice, the CG position is related to the aircraft by

$$
CG = \frac{W \cdot (BA - RA)}{C} + K,
$$
where $W$ is the actual aircraft weight, $BA$ is the horizontal distance from the reference datum to the CG location, $RA$ is the reference arm, $C$ is the aircraft weight constant, and $K$ is a constant introduced to avoid negative index values.

**CG gap:** The CG gap is computed as the absolute deviation normalized by the optimal value:
$$
\text{Gap}(\%) = \frac{|CG_{actual} - CG_{opt}|}{|CG_{opt}|} \times 100
$$



**Algorithm runtime:** Algorithm runtime refers to the wall-clock execution time of each baseline from the start of optimization to termination under the prescribed time limit.



## 4.1 Multi-constraint cargo loading

This workload evaluates air cargo loading algorithms under **incrementally modeling complexity sets**.  
It supports **A320 (narrow-body, bulk hold)** and **B777 (wide-body, ULD hold)**, and reports performance across multiple constraint levels.

#### Modeling and solver notes
All experiments in this subsection use an assignment-based optimization model. Let $I$ be the set of cargo items and $H$ the set of cargo holds. The binary decision variable

$$
x_{ih} =
\begin{cases}
1, & \text{if cargo item } i \text{ is assigned to hold } h, \\
0, & \text{otherwise}.
\end{cases}
$$

The total loaded weight and longitudinal CG surrogate are computed as

$$
W = W_0 + \sum_{i \in I}\sum_{h \in H} w_i x_{ih},
\qquad
CG = CG_0 + \sum_{i \in I}\sum_{h \in H} w_i a_h x_{ih},
$$

where $W_0$ and $CG_0$ are the aircraft initial zero-fuel quantities loaded from the aircraft profile, $w_i$ is the cargo weight, and $a_h$ is the hold CG coefficient. The CG envelope is interpolated from `stdZfw_a.csv` and `stdZfw_f.csv`, and the target CG used by the code is

$$
CG^\star(W) = CG^{aft}(W) + \frac{CG^{fwd}(W) - CG^{aft}(W)}{3}.
$$

The common single-segment objective is

$$
\min \; |CG - CG^\star(W)|.
$$

The main constraints are

$$
\sum_{h \in H} x_{ih} \le 1, \qquad \forall i \in I,
$$

$$
\sum_{i \in I} w_i x_{ih} \le \bar W_h, \qquad \forall h \in H,
$$

$$
CG^{aft}(W) \le CG \le CG^{fwd}(W),
$$

plus experiment-specific operational constraints such as ULD-type compatibility, exclusive holds, continuous loading, loading order, and dangerous-goods isolation. The released baselines are implemented in `algorithm/for_narrow/` and `algorithm/for_wide/` and are executed with a unified time limit through the Python scripts below. In the codebase, `MILP`, `MINLP`, and `QP` are custom mathematical-programming-style baselines, while `DP`, `CP`, `GA`, `PSO`, `CS`, `ACO`, `ABC`, and `MBO` are exact-search or heuristic baselines evaluated under the same interface.

### Experiment 1: Constraint incremental analysis

#### Prerequisites
Prepare the following paths:
- **code root path**: a directory that contains `algorithm/` and `multi_constraint_cargo_loading/`
- **aircraft path**: aircraft configuration files (default: `G:\AirCa\code\aircraft_data`)
- **cargo path**: flight cargo CSV files
- **output path**: where results will be saved

> Tip: run the script from `multi_constraint_cargo_loading/` to avoid import issues.

#### Key arguments
- `--mode`: `both` (A320 + B777), `narrowbody` (A320 only), `widebody` (B777 only)
- `--aircraft`: aircraft type list for narrow-body mode (e.g., `A320`)
- `--n-flights`: number of flights sampled per aircraft
- `--time-limit`: per-algorithm time limit (seconds)
- `--code-path`, `--benchmark-path`, `--cargo-data-path`, `--output-path`: paths for reproducibility

#### Constraint levels reproduced by the code
For **narrow-body aircraft**, the script evaluates three nested levels:

- **Level 1 (Cargo Only)**: per-hold weight and CG objective.
- **Level 2 (Cargo + Hold)**: Level 1 plus exclusive-hold constraints and ULD/cargo-type compatibility.
- **Level 3 (Full Constraints)**: Level 2 plus loading-order constraints, continuous-loading constraints, and dangerous-goods isolation.

For **wide-body aircraft**, the script uses `loose`, `medium`, and `tight` settings:

- `loose`: relaxed hold weight limits and relaxed CG envelope, without exclusivity/capacity/ULD-type checks.
- `medium`: original hold limits with capacity, exclusivity, and ULD-type checks.
- `tight`: stricter hold limits, full hard constraints, and explicit CG envelope feasibility checks.

Mathematically, the additional constraints can be written as

$$
\sum_{i \in I} x_{ih} \le 1, \qquad \forall h \in H \text{ (wide-body ULD capacity)},
$$

$$
x_{ih} = 0, \qquad \forall (i,h)\text{ with incompatible cargo/ULD type},
$$

$$
\sum_{i \in I} x_{ih} + \sum_{i \in I} x_{ik} \le 1,
\qquad \forall (h,k) \in \mathcal{E},
$$

where $\mathcal{E}$ is the set of mutually exclusive hold pairs. The narrow-body full model further forbids destination mixing inside the same hold and enforces adjacency-based dangerous-goods isolation.

#### Run
**Recommended (run both A320 and B777 with defaults):**
```bash
cd <code root path>\multi_constraint_cargo_loading
python Constraint_incremental_analysis.py --mode both

# A320 only
python Constraint_incremental_analysis.py --mode narrowbody --aircraft A320 --n-flights 10

# B777 only
python Constraint_incremental_analysis.py --mode widebody --n-flights 50

# Full template
python Constraint_incremental_analysis.py ^
  --code-path <code root path> ^
  --benchmark-path <aircraft path> ^
  --cargo-data-path <cargo path> ^
  --output-path <output path> ^
  --mode both ^
  --n-flights 10 ^
  --time-limit 30
```



### Experiment 2: Aircraft configuration comparison (A320 / B777 / C919)

This experiment compares three representative aircraft types—**A320**, **B777**, and **C919**—under the same algorithm set.  
For each aircraft, it reports the **CG gap** and **computation time** achieved by each algorithm, highlighting how aircraft configurations affect optimization difficulty.

#### What you need to prepare
- **code root**: the project root that contains the `algorithm/` package
- **cargo path**: a folder containing `BAKFLGITH_LOADDATA*.csv`
- **aircraft path**: aircraft configuration data folder (contains subfolders like `A320/`, `B777/`, `C919/`)
- **output path**: where results will be saved

#### Key settings
- `--aircraft all`: run A320 + B777 + C919 in one run  
- `--mode single`: batch evaluation on *single-segment* flights  
- `--n-flights`: number of flights sampled per aircraft  
- `--time-limit`: per-algorithm time limit (seconds, optional)  
- `--algos`: comma-separated algorithm names (e.g., `MILP,GA,PSO`), or `all`

#### Optimization model used in this comparison
This experiment keeps the same single-segment objective

$$
\min \; |CG - CG^\star(W)|
$$

across aircraft types and changes only the aircraft-specific input data: hold layout, hold capacity, exclusive-hold relations, allowed ULD types, and interpolated CG envelope. This design makes the reported differences attributable to aircraft configuration rather than to a change of model or evaluation metric. For exact baselines, the code constructs surrogate optimization matrices internally:

- `MILP`: linear cost matrix with assignment and hold-capacity constraints.
- `MINLP`: nonlinear surrogate with squared CG deviation.
- `QP`: quadratic surrogate of the CG-deviation term.

The heuristic baselines optimize the same problem through the common `evaluate_solution` / `get_objective_value` interface, so all methods are compared under the same feasibility checks and time budget.

#### Run
```bash
python "aircraft configuration comparison script" --code-root "code root" --cargo-data-dir "cargo path" --aircraft-data-dir "aircraft path" --output-dir "output path" --aircraft all --mode single --n-flights 100 --time-limit 30 --algos all
```




## 4.2 Cargo loading with massive variables


### Experiment 3: Variable scaling analysis (A320)

This experiment studies scalability by detaching bulk cargo into progressively finer granularities (100/50/25/10 kg), which increases the number of decision variables from tens to thousands per flight. For each scale, it records computation time and CG gap of multiple combinatorial optimization algorithms to reveal their scaling behavior.

#### What you need to prepare
- **code root**: a directory that contains the `algorithm/` folder (the script imports `algorithm.for_narrow.*`).
- **aircraft path**: aircraft configuration files (default: `G:\AirCa\code\aircraft_data`) and it must include `A320.csv` (the 2nd column is used as `hold_id`) plus optional CG limit files `stdZfw_a.csv/stdZfw_f.csv`.
- **cargo path**: directory containing `BAKFLGITH_LOADDATA*.csv`.
- **output path**: where result CSV files will be saved.

#### Key settings
- `--split-thresholds`: cargo detaching thresholds in kg (default: `100,50,25,10`)  
- `--n-flights`: number of top single-segment flights tested (default: `4`)  
- `--time-limit`: time limit passed to each algorithm (default: `120` seconds)  
- `--algo-timeout`: hard timeout per algorithm run (default: `120` seconds)  
- `--constraint-level`: `basic` or `tight` (if `tight`, CG envelope violations are also checked and reported)  
- `--no-exclusive-check`: disable exclusive-hold violation checking (optional)  
- `--algorithms`: optionally run a subset of algorithms by class name (comma-separated)

#### Modeling details for reproducibility
After cargo splitting, each detached piece is treated as an independent item in the same narrow-body assignment model:

$$
\min \; |CG - CG^\star(W)|
$$

subject to

$$
\sum_{h \in H} x_{ih} \le 1, \qquad \forall i \in I,
$$

$$
\sum_{i \in I} w_i x_{ih} \le \bar W_h, \qquad \forall h \in H,
$$

and, when enabled, the exclusive-hold and CG-envelope constraints. The only controlled factor changed in this experiment is the split threshold, which changes $|I|$ and therefore the number of binary assignment decisions. The released script uses the same algorithm list and the same timeout wrapper for every threshold, which makes the scaling trend directly reproducible.

For the mathematical-programming baselines in `algorithm/for_narrow/exact_algorithms1.py`, the surrogate objectives implemented in code are:

$$
\texttt{MILP: } \min \sum_{i \in I}\sum_{h \in H} |1000a_h - CG^\star|\, w_i x_{ih},
$$

$$
\texttt{MINLP: } \min \sum_{i \in I}\sum_{h \in H} (1000a_h - CG^\star)^2 w_i^2 x_{ih},
$$

$$
\texttt{QP: } \min \frac{1}{2}x^\top Qx + c^\top x,
$$

where the diagonal entries of $Q$ and the vector $c$ are built from the same CG-deviation terms in the released code.

#### Run
```bash
python "code_root/AirCa/code/cargo_loading_with_massive_variables/Variable_scaling_analysis.py" --code-root "code root" --aircraft-data-dir "aircraft path" --cargo-data-dir "cargo path" --output-dir "output path" --n-flights 4 --time-limit 120 --algo-timeout 120 --split-thresholds 100,50,25,10 --constraint-level basic
```

### Experiment 4: Timeout behavior characterization (A320)

This experiment evaluates how solution quality changes under strict time budgets by imposing time limits of **10s, 30s, 60s, and 120s**. It measures the **CG gap at termination** for each algorithm, distinguishing methods that converge quickly from those that need longer computation.

#### What you need to prepare
- **code root**: a directory that contains the `algorithm/` folder (the script imports `algorithm.for_narrow.*`).
- **aircraft path**: aircraft configuration files (must include `A320.csv`; the 2nd column is used as `hold_id`, and optional CG limit files may be used).
- **cargo path**: directory containing `BAKFLGITH_LOADDATA*.csv`.
- **output path**: where result CSV files will be saved.

#### Key settings
- `--time-limits`: time limits (seconds) swept in this experiment (use `10,30,60,120`).
- `--split-threshold`: cargo detaching threshold in kg used to fix the variable scale (e.g., `50`).
- `--n-flights`: number of top single-segment flights tested.
- `--extra-timeout-buffer`: extra seconds added to avoid hard cutoff (recommended).
- `--algorithms`: optionally run a subset of algorithms by class name (comma-separated); otherwise run all.

#### Optimization model and stopping rule
The optimization model is exactly the same as in Experiment 3; only the stopping budget changes. Let $T$ denote the wall-clock limit passed to each baseline. The script records the best solution returned before timeout and reports its terminal CG gap:

$$
\text{Gap}(T) = \frac{|CG(T) - CG^\star(W(T))|}{|CG^\star(W(T))|} \times 100\%.
$$

This experiment is therefore a time-to-quality benchmark rather than a different optimization model. The `--extra-timeout-buffer` argument is used only to prevent premature thread termination outside the algorithm itself and does not change the underlying objective or feasibility rules.

#### Run
```bash
python "code_root/AirCa/code/cargo_loading_with_massive_variables/Timeout_behavior_characterization.py" --code-root "code root" --aircraft-data-dir "aircraft path" --cargo-data-dir "cargo path" --output-dir "output path" --n-flights 4 --split-threshold 50 --time-limits 10,30,60,120 --extra-timeout-buffer 30
```


## 4.3 Multi-segment cargo loading


### Experiment 5: Multi-stage trade-off analysis (A320 & B777)

This experiment evaluates multi-stage cargo loading under two objective modes: **CG-priority** (minimize CG deviation) and **profit-priority** (maximize transportation profit while satisfying CG envelope feasibility). Both modes use the same multi-stage sequential constraint (long-haul cargo is loaded first into inner holds), and we report the achieved **CG gap** and **profit** on **A320** and **B777**.

#### What you need to prepare
- **code root**: a directory that contains the `algorithm/` folder (the script loads narrow-body and wide-body algorithms from different modules).
- **aircraft path**: aircraft configuration data (must include subfolders/files for `A320` and `B777`).
- **cargo path**: directory containing `BAKFLGITH_LOADDATA*.csv`.
- **output path**: where result CSV files will be saved.

#### Key settings
- `--aircraft A320 B777`: run both aircraft types (recommended).
- `--n-flights`: number of flights sampled per aircraft.
- `--time-limit`: per-algorithm time limit (seconds).
- Objective modes are evaluated automatically:
  - **CG-priority**: strictly minimizes CG gap.
  - **profit-priority**: maximizes profit with CG envelope as feasibility.

#### Multi-objective model used by the released code
This experiment uses the class `CargoLoadingProblemMultiStage` in `multi_segment_cargo_loading/Multi-stage_trade-off_analysis.py`. The aircraft state is still computed from

$$
W = W_0 + \sum_{i \in I}\sum_{h \in H} w_i x_{ih},
\qquad
CG = CG_0 + \sum_{i \in I}\sum_{h \in H} w_i a_h x_{ih},
$$

but the objective is multi-objective. First, the script computes cargo revenue with the released piecewise tariff table:

$$
R_i = \max\{70,\; r(w_i)\, w_i\},
$$

where

$$
r(w_i)=
\begin{cases}
12.66, & w_i \le 44, \\
9.74, & 45 \le w_i \le 99, \\
9.07, & 100 \le w_i \le 299, \\
7.16, & 300 \le w_i \le 499, \\
6.27, & 500 \le w_i \le 999, \\
5.44, & w_i \ge 1000.
\end{cases}
$$

Gross revenue and profit are then

$$
R^{gross} = \sum_{i \in I}\sum_{h \in H} R_i x_{ih},
$$

$$
R^{profit} = R^{gross}\left(1 - 0.5\left(1-e^{-\text{Gap}/50}\right)\right),
$$

where $\text{Gap}$ is the CG gap percentage. The optimization objective used in code is

$$
\min \; \alpha \cdot \frac{\text{Gap}}{100} - \beta \cdot \frac{R^{profit}}{R_{\max}},
$$

where $\alpha=\texttt{cg\_weight}$, $\beta=\texttt{revenue\_weight}$, and $R_{\max}$ is the estimated maximum achievable revenue used for normalization. The same feasibility constraints as the single-stage model remain active, and for wide-body aircraft the code additionally enforces one ULD per hold and ULD-type compatibility.

#### Run
```bash
python "multi-stage trade-off script" --benchmark-path "aircraft path" --cargo-data-path "cargo path" --output-path "output path" --aircraft A320 B777 --n-flights 10 --time-limit 15
```


### Experiment 6: Single-stage versus multi-stage comparison (A320 & B777)

This experiment evaluates **150 multi-segment flights** using both **single-stage** and **multi-stage** optimization. It compares **CG gap**, **transportation profit**, and **computation time** on these multi-segment instances to quantify the benefits of multi-stage optimization.

#### What you need to prepare
- **code root**: a directory that contains the `algorithm/` folder (narrow-body and wide-body algorithms are loaded from different modules).
- **aircraft path**: aircraft configuration data (must include `A320` and `B777` configurations).
- **cargo path**: directory containing `BAKFLGITH_LOADDATA*.csv`.
- **output path**: where result CSV files will be saved.

#### Key settings
- `--aircraft A320 B777`: run both aircraft types (recommended).
- `--n-pairs`: number of multi-segment flight instances sampled for the comparison (set to `150` for this experiment).
- `--time-limit`: per-algorithm time limit (seconds).

#### What changes between single-stage and multi-stage
The underlying objective and feasibility checks are the same as in Experiment 5. The difference lies in how the assignment decisions are executed:

- **Single-stage**: all segment cargo is optimized in one joint decision.
- **Multi-stage**: the loading plan is optimized sequentially by stage/segment, so earlier-stage decisions constrain later-stage feasible assignments.

In both cases, the reported metrics are computed from the final complete loading plan:

$$
\text{CG gap (\%)} = \frac{|CG - CG^\star(W)|}{|CG^\star(W)|}\times 100\%,
\qquad
\text{profit} = R^{profit}.
$$

This makes the comparison reproducible because both methods are evaluated by the same post-hoc `evaluate_solution` routine; only the decision process differs.

#### Run
```bash
python "single-vs-multi-stage comparison script" --benchmark-path "aircraft path" --cargo-data-path "cargo path" --output-path "output path" --aircraft A320 B777 --n-pairs 150 --time-limit 15
```

## 4.4 Running example
```
python "<code root path>/AirCa/code/multi_constraint_cargo_loading/Aircraft_configuration_comparison..py" --code-root "code root" --cargo-data-dir "cargo path" --aircraft-data-dir "aircraft path" --output-dir "output path" --aircraft B777 --mode demo --flight-number "flight number" --algos all --time-limit 120

```

![alt text](image-2.png)


# 5. The AirCa APIs
In addition to our AirCa dataset, we release the AirCa package,
including three types of APIs. It is designed to faciliate researchers
in developing aircraft cargo loading applications.The details are
presented as follows:

**DataDownloader**. This API allows researchers to download the
AirCa data. the code presents how to utilize the DataDownloader
API to download the up-to-date AirCa data.
DataDownloader. This API allows researchers to download the
AirCa data. Figure 4 presents how to utilize the DataDownloader
API to download the up-to-date AirCa data.
```python
from api . download_airca import AirCaDownloader
downloader = AirCaDownloader ()
# Download data A320
downloader . download_AirCa ( url , path , aircraft_type =" A320 " ,
date =" 2024 -10 -12 ")
# Download data B737
downloader . download_AirCa ( url , path , aircraft_type =" B737 " ,
date = None )
# Download data for all available aircraft types
downloader . download_AirCa ( url , path , aircraft_type = None ,
date = None )
```
  
**DataRetriever**. This API enables researchers to conveniently
obtain the AirCa data stroed in the local machine. For instance,
the code shows how to employ the DataRetriever API to obtain the
AirCa data for aircraft type B777.
```python
from api . retriever import Retriever
retriever = Retriever ()
# Enter the type A320
retriever . retrieve ( path = path , aircraft_type = " A320 ")
# Enter the type B777
retriever . retrieve ( path = path , aircraft_type = " B777 ")
# Enter the type B787
retriever . retrieve ( path = path , aircraft_type = " B787 ")
```

**DataLoader**. This API is designed to assist researchers in their
applications of aircraft cargo loading. It allows researchers to flex-
ibly and seamlessly merge multiple modalities of AirCa data. It
exposes the AirCa through a DataLoader object after performing necessary data preprocessing techniques. A PyTorch example of
using our DataLoader API for training DNNs is shown in the code.
```python
import torch
from torch . utils . data import DataLoader
# generate AirCa ( A320 ) dataset for training
dataloader1 = DataLoader ( AircraftDataset ( path ," A320 ") ,
batch_size = batch_size , shuffle = True )
# generate AirCa ( B777 ) dataset for training
dataloader2 = DataLoader ( AircraftDataset ( path ," B777 ") ,
batch_size = batch_size , shuffle = True )
# generate AirCa ( B787 ) dataset for training
dataloader3 = DataLoader ( AircraftDataset ( path ," B787 ") ,
batch_size = batch_size , shuffle = True )
train_model ( dataloader1 , baseline_name , criterion ,
optimizer , epochs =600)
```

# 6. References
[1] Zhao, X., Dong, Y., & Zuo, L. (2023). A combinatorial optimization approach for air cargo palletization and aircraft loading. *Mathematics, 11*(13), 2798.  
[2] Mesquita, A. C. P., & Sanches, C. A. A. (2024). Air cargo load and route planning in pickup and delivery operations. *Expert Systems with Applications, 249*, 123711.  
[3] Yan, S., Lo, C.-T., & Shih, Y.-L. (2006). Cargo container loading plan model and solution method for international air express carriers. *Transportation Planning and Technology, 29*(6), 445–470.  
[4] Yan, S., Shih, Y.-L., & Shiao, F.-Y. (2008). Optimal cargo container loading plans under stochastic demands for air express carriers. *Transportation Research Part E: Logistics and Transportation Review, 44*(3), 555–575.  
[5] Limbourg, S., Schyns, M., & Laporte, G. (2012). Automatic aircraft cargo load planning. *Journal of the Operational Research Society, 63*(9), 1271–1283.  
[6] Zhao, X., Yuan, Y., Dong, Y., & Zhao, R. (2021). Optimization approach to the aircraft weight and balance problem with the centre of gravity envelope constraints. *IET Intelligent Transport Systems, 15*(10), 1269–1286.  
[7] Lurkin, V., & Schyns, M. (2015). The airline container loading problem with pickup and delivery. *European Journal of Operational Research, 244*(3), 955–965.  
[8] Zhu, L., Wu, Y., Smith, H., & Luo, J. (2023). Optimisation of containerised air cargo forwarding plans considering a hub consolidation process with cargo loading. *Journal of the Operational Research Society, 74*(3), 777–796.  
[9] Chenguang, Y., Liu, H., & Yuan, G. (2018). Load planning of transport aircraft based on hybrid genetic algorithm. In *MATEC’18, Vol. 179* (pp. 01007). EDP Sciences.  
[10] Dahmani, N., & Krichen, S. (2016). Solving a load balancing problem with a multi-objective particle swarm optimisation approach: application to aircraft cargo transportation. *International Journal of Operational Research, 27*(1-2), 62–84.  
[11] Dahmani, N., & Krichen, S. (2013). On solving the bi-objective aircraft cargo loading problem. In *ICMSAO’13* (pp. 1–6). IEEE.  
[12] Gajda, M., Trivella, A., Mansini, R., & Pisinger, D. (2022). An optimization approach for a complex real-life container loading problem. *Omega, 107*, 102559.  


<!-- If you find this helpful, please cite our paper:

```shell
@misc{wu2023lade,
      title={LaDe: The First Comprehensive Last-mile Delivery Dataset from Industry}, 
      author={Lixia Wu and Haomin Wen and Haoyuan Hu and Xiaowei Mao and Yutong Xia and Ergang Shan and Jianbin Zhen and Junhong Lou and Yuxuan Liang and Liuqing Yang and Roger Zimmermann and Youfang Lin and Huaiyu Wan},
      year={2023},
      eprint={2306.10675},
      archivePrefix={arXiv},
      primaryClass={cs.DB}
}
``` -->
