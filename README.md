# AirCa

 

# Contents

- [1. About Dataset](#1-about-dataset)
- [2. Download](#2-download)
- [3. Description](#3-description)
  * [3.1 AirCa-W](#31-airca-w)
  * [3.2 AirCa-N](#32-airca-n)
  * [3.3 Constraints description](#33-constraints-description)
- [4. Leaderboard](#4-leaderboard)
  * [4.1 Long-term Cargo Capacity Prediction](#41-long-term-cargo-capacity-prediction)
  * [4.2 Optimization of Cargo Loading](#42-optimization-of-cargo-loading)
  * [4.3 Cargo balancing/loading with Large Language Model optimization](#43-cargo-balancingloading-with-large-language-model-optimization)
- [5. The AirCa APIs](#5-the-airca-apis)
- [6. References](#5-references)
- [7. Citations](#6-citation)


Dataset Download: https://huggingface.co/datasets/LINC-BIT/AirCa       
Dataset Website:  https://huggingface.co/datasets/LINC-BIT/AirCa   
Code Link: https://github.com/LINC-BIT/AirCa     
Paper Link:  


# 1. About Dataset
**AirCa** is a publicly available  aircraft cargo loading dataset with  millions of instances from industry. It has three unique characteristics: 
(1) Large-scale, AirCa contains in total 6,071k records and 1,092k flights, covering 6 aircraft types and 425 airports over a total span of 9 months. 
(2) Comprehensive information, AirCa is delivered to provide rich information pertaining to aircraft cargo loading, including detailed cargo characteristic information, loading-event logs, flight destination, and comprehensive loading constraints in practical scenarios.
(3) Diversity,  AirCa aims to increase data diversity from three perspectives: destination diversity, Flight diversity, and Constraint diversity.



![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65d6eb2651e148d01873ec81/8lPaRQm5J7z6tkRqMJrfc.jpeg)

The figure depicts the process of air cargo loading, starting with terminal administration, where goods are processed and prepared for transportation.
Cargo loading follows as goods are transferred to the aircraft, and then flight preparation and flying take place as the plane gets ready for departure. 
The cargo is carefully organized in the Unit Load Devices (ULD), which are containers or pallets used to carry the cargo efficiently. 
For wide-body aircraft cargo holds, like the B777, there are designated areas for both small ULD containers and larger pallets. 
Meanwhile, narrow-body aircraft cargo holds, like the A320, have a different arrangement suited for smaller loads.
The cargo types include bulk cargo and special goods, which require specific handling due to their size, fragility, or value.
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
Below is the detailed field of each sub-dataset.
## 3.1 AirCa-W
| Data field              | Description                                           | Unit/format           |
|-------------------------|-------------------------------------------------------|-----------------------|
| **Cargo information**    |                                                       |                       |
| Loading order           | Record of the cargo loading order                    | String                |
| ID                      | Unique identifier for ULD                           | ID                    |
| Weight                  | Weight of ULD                                        | String                |
| ULD type                | Types of ULD include general cargo, special cargo    | String                |
| Priority                | Cargo loading priority                               | String                |
| Length                  | Length of ULD                                        | Float                 |
| Width                   | Width of ULD                                         | Float                 |
| Height                  | Height of ULD                                        | Float                 |
| Transship cargo         | The record of whether it is transship cargo          | Bool                  |
| **Flight information**  |                                                       |                       |
| Loading time            | Record of the cargo loading time                     | Time                  |
| Flight ID (anonymity)   | Record of the different flights                      | ID                    |
| Destination airport     | Record of the airport's name                         | String                |
| Segment                 | The record of whether it is multi-segment flight     | Bool                  |
| **Aircraft information**|                                                       |                       |
| Aircraft type           | The type of the aircraft                             | String                |
| Constraints             | The constraints of air cargo loading                 | Constraint format     |

## 3.2 AirCa-N

| Data field              | Description                                           | Unit/format           |
|-------------------------|-------------------------------------------------------|-----------------------|
| **Cargo information**    |                                                       |                       |
| ID                      | Unique identifier for ULD                           | ID                    |
| Weight                  | Weight of cargo                                      | String                |
| Bulk type               | Types of bulk include general cargo, special cargo  | String                |
| Priority                | Cargo loading priority                               | String                |
| Volume                  | The volume of the cargo                              | Float                 |
| **Flight information**  |                                                       |                       |
| Loading time            | Record of the cargo loading time                     | Time                  |
| Flight ID (anonymity)   | Record of the different flights                      | ID                    |
| Destination airport     | Record of the airport's name                         | String                |
| Segment                 | The record of whether it is multi-segment flight     | Bool                  |
| **Aircraft information**|                                                       |                       |
| Aircraft type           | The type of the aircraft                             | String                |
| Constraints             | The constraints of air cargo loading                 | Constraint format     |

## 3.3 Constraints description
| Constraint                            | Description                                                                                  | Unit/format           |
|---------------------------------------|----------------------------------------------------------------------------------------------|-----------------------|
| **Cargo constraints**                 |                                                                                              |                       |
| Special cargo space weight constraint | This constraint defines the maximum allowable weight of special cargo in the space           | Float                 |
| Dangerous cargo isolation constraint  | Any two special cargo loading locations need to maintain a specified distance                | String                |
| **Aircraft cargo hold constraints**   |                                                                                              |                       |
| ULD correspondence constraint         | Get the corresponding relationship of cargo types and verify each piece of cargo data        | String                |
| ULD Type Restriction Rules            | If the container type in the loading data is not one of the ones defined in ULD Type, the check fails | Bool             |
| ULD type and ULD number constraint    | If the container type does not correspond to the container serial number, the verification fails | Bool            |
| Cargo hold availability constraint    | Before loading, check whether the cargo hold is available                                     | String                |
| Mixed cargo space constraint          | Check whether there is mixed loading in the cargo hold                                       | Bool                  |
| Number of ULD constraint              | The quantity of ULD cannot exceed this specified value                                       | Float                 |
| Front/Rear compartment constraint     | Ensure weight in the front (FWD) and rear (AFT) compartments do not exceed the defined limits. | Float                 |
| Cargo Type validity constraint        | Check whether cargo type is valid and belongs to predefined cargo types.                     | String                |
| **Loading constraints**               |                                                                                              |                       |
| Weight constraint                     | Maximum load weight of the cargo hold                                                        | Float                 |
| CG constraint                         | Ideal center of gravity range for airliner when zero fuel                                    | Float                 |
| Volume constraint                     | The volume of cargo cannot exceed this specified value                                        | Float                 |
| Joint weight constraint               | Total load weight constraints for multiple cargo holds                                       | Float                 |
| Cargo space weight constraint         | This constraint defines the maximum weight limit for a cargo space.                          | Float                 |
| Continuous loading constraint         | Some types of ULDs need to be loaded according to the load sequence                          | String                |
| Load order constraint                 | Goods must be loaded in the specified order                                                   | String                |

# 4. Leaderboard
Blow shows the performance of different methods in AirCa.
## 4.1 Long-term Cargo Capacity Prediction

<!-- Experimental results of Long-term Cargo Capacity Prediction. We use bold and underlined fonts to denote the best and runner-up model, respectively. -->

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65d6eb2651e148d01873ec81/wcNpHRQ-eVLtufvsH8Gpz.png)



## 4.2 Optimization of Cargo Loading

Experimental results of Optimization of Cargo Loading. The introduction of 12 baselines is shown as follows:
- **COM** [1]: Combinatorial Optimization Model solves discrete optimization tasks by searching for an optimal arrangement among a finite set of feasible solutions.
- **IOM** [2]: Improved Combinatorial Optimization Model obtains better solutions for discrete optimization tasks by refining search strategies to more effectively explore feasible configurations.
- **NL-CPLEX** [3]: NL-CPLEX addresses nonlinear optimization tasks by leveraging branch-and-bound and cutting-plane techniques to efficiently explore the solution space.
- **SDCCLPM** [4]: Stochastic-Demand Cargo Container Loading Plan Model optimizes container loading configurations under demand uncertainty by incorporating probabilistic approaches to balance capacity and cost requirements.
- **MLIP** [5]: Mixed Integer Linear Program finds optimal solutions to discrete optimization problems by combining integer constraints with linear relationships in a branch-and-bound search process.
- **MLIP-WBP** [6]: MLIP-WBP optimizes weighted bin packing by employing a Mixed Integer Linear Programming formulation to balance item distribution and capacity constraints.
- **MLIP-ACLPDD** [7]: MLIP-ACLPDD solves advanced cargo loading planning under uncertain demand by incorporating robust constraints into a Mixed Integer Linear Programming framework.
- **HGA** [8]: Hybrid Genetic Algorithm enhances solution quality by combining evolutionary operators with complementary search techniques to accelerate convergence and explore the solution space more thoroughly.
- **GA-normal** [9]: GA-normal employs foundational genetic algorithm operations—selection, crossover, and mutation—to explore solutions within a population-based search framework.
- **DMOPSO** [10]: Discrete Multi-Objective Particle Swarm Optimization locates Pareto-optimal solutions in discrete search spaces by adapting swarm-based velocity and position update mechanisms to address multiple conflicting objectives.
- **PSO-normal** [11]: PSO-normal employs the basic velocity and position update rules, guided by personal and global best solutions, to iteratively converge on an optimal search space configuration.
- **RCH** [12]: Randomized Constructive Heuristic incrementally constructs feasible solutions by integrating stochastic choices during each step, thus diversifying the search process and enhancing solution discovery.

| **Method**       | **B777** MAC(%)↓ | **B777** INDEX(%)↓ | **B777** TIME(s)↓ | **A320** MAC(%)↓ | **A320** INDEX(%)↓ | **A320** TIME(s)↓ | **B787** MAC(%)↓ | **B787** INDEX(%)↓ | **B787** TIME(s)↓ |
|------------------|------------------|--------------------|-------------------|------------------|--------------------|-------------------|------------------|--------------------|-------------------|
| **COM**          | 23.93 ± 0.59     | 3.40 ± 1.64        | 0.06 ± 0.04       | 21.14 ± 0.28     | 6.46 ± 2.20        | 0.06 ± 0.05       | 23.71 ± 0.47     | 3.10 ± 1.58        | 0.03 ± 0.03       |
| **IOM**          | 23.90 ± 0.59     | 3.40 ± 1.62        | 0.07 ± 0.08       | 21.16 ± 0.28     | 6.50 ± 2.16        | 0.07 ± 0.05       | 23.71 ± 0.46     | 3.08 ± 1.56        | 0.06 ± 0.05       |
| **NL-CPLEX**     | 23.92 ± 0.58     | 3.45 ± 1.60        | 0.08 ± 0.06       | 21.15 ± 0.29     | 6.48 ± 2.18        | 0.08 ± 0.07       | 23.70 ± 0.47     | 3.07 ± 1.61        | 0.05 ± 0.04       |
| **SDCCLPM**      | 23.91 ± 0.59     | 3.40 ± 1.63        | 0.07 ± 0.05       | 21.15 ± 0.28     | 6.46 ± 2.18        | 0.07 ± 0.06       | 23.70 ± 0.46     | 3.08 ± 1.57        | 0.05 ± 0.04       |
| **MLIP**         | 23.92 ± 0.57     | 3.47 ± 1.59        | 0.06 ± 0.07       | 21.14 ± 0.29     | 6.45 ± 2.20        | 0.06 ± 0.05       | 23.69 ± 0.46     | 3.04 ± 1.63        | 0.03 ± 0.02       |
| **MLIP-WBP**     | 23.92 ± 0.58     | 3.45 ± 1.60        | 3.53 ± 5.78       | 21.15 ± 0.29     | 6.47 ± 2.19        | 1.43 ± 0.78       | 23.70 ± 0.47     | 3.07 ± 1.61        | 1.43 ± 0.85       |
| **MLIP-ACLPDD**  | 23.93 ± 0.59     | 3.44 ± 1.65        | 3.46 ± 1.61       | 21.14 ± 0.29     | 6.44 ± 2.20        | 1.46 ± 0.98       | 23.71 ± 0.47     | 3.12 ± 1.60        | 1.67 ± 1.02       |
| **HGA**          | 23.37 ± 0.47     | 3.23 ± 1.06        | 253.30 ± 0.80     | 21.14 ± 0.22     | 6.69 ± 1.80        | 1.80 ± 0.84       | 23.46 ± 0.24     | 3.86 ± 1.74        | 193.62 ± 0.51     |
| **GA-normal**    | 23.35 ± 0.48     | 3.13 ± 1.08        | 221.82 ± 0.52     | 21.14 ± 0.22     | 6.71 ± 1.80        | 1.81 ± 0.51       | 23.44 ± 0.23     | 3.73 ± 1.69        | 145.70 ± 0.17     |
| **DMOPSO**       | 23.12 ± 0.49     | 1.56 ± 1.65        | 266.11 ± 2.61     | 21.10 ± 0.28     | 6.59 ± 2.43        | 2.60 ± 0.61       | 23.29 ± 0.29     | 3.00 ± 2.39        | 204.13 ± 2.02     |
| **PSO-normal**   | 23.19 ± 0.44     | 2.13 ± 1.81        | 211.73 ± 2.70     | 21.09 ± 0.28     | 6.56 ± 2.43        | 2.61 ± 0.70       | 23.30 ± 0.27     | 3.09 ± 2.19        | 199.24 ± 1.80     |
| **RCH**          | 23.35 ± 0.50     | 3.23 ± 1.23        | 200.63 ± 0.06     | 21.07 ± 0.24     | 6.55 ± 1.93        | 1.78 ± 0.06       | 23.41 ± 0.26     | 3.50 ± 1.93        | 200.20 ± 0.02     |



| **Method**         | **Segment 1 MAC(%)↓** | **Segment 1 INDEX(%)↓** | **Segment 1 TIME(s)↓** | **Segment 2 MAC(%)↓** | **Segment 2 INDEX(%)↓** | **Segment 2 TIME(s)↓** |
|--------------------|-----------------------|-------------------------|------------------------|-----------------------|-------------------------|------------------------|
| **COM**            | 23.59 ± 0.40          | 2.72 ± 1.56             | 0.73 ± 0.61            | 24.29 ± 0.74          | 3.89 ± 2.32             | 1.25 ± 0.92            |
| **IOM**            | 23.65 ± 0.41          | 3.02 ± 1.62             | 1.19 ± 0.90            | 24.30 ± 0.73          | 4.02 ± 2.42             | 1.82 ± 1.19            |
| **NL-CPLEX**       | 23.61 ± 0.41          | 2.65 ± 1.49             | 1.06 ± 0.94            | 24.30 ± 0.74          | 3.88 ± 2.27             | 1.96 ± 1.39            |
| **SDCCLPM**        | 23.63 ± 0.41          | 2.96 ± 1.61             | 1.11 ± 0.95            | 24.28 ± 0.74          | 3.97 ± 2.38             | 1.81 ± 1.35            |
| **MLIP**           | 23.63 ± 0.42          | 2.68 ± 1.48             | 0.84 ± 0.75            | 24.28 ± 0.74          | 3.87 ± 2.24             | 1.21 ± 0.85            |
| **MLIP-WBP**       | 23.61 ± 0.41          | 2.65 ± 1.49             | 32.06 ± 22.02          | 24.30 ± 0.74          | 3.88 ± 2.27             | 44.32 ± 22.77          |
| **MLIP-ACLPDD**    | 23.60 ± 0.40          | 2.73 ± 1.54             | 34.05 ± 22.46          | 24.28 ± 0.74          | 3.88 ± 2.33             | 51.55 ± 27.58          |
| **HGA**            | 23.44 ± 0.23          | 3.73 ± 1.69             | 36.10 ± 8.55           | 23.39 ± 0.30          | 3.32 ± 2.15             | 23.86 ± 2.69           |
| **GA-normal**      | 23.43 ± 0.24          | 3.65 ± 1.77             | 28.70 ± 3.45           | 23.25 ± 0.24          | 2.37 ± 1.74             | 23.57 ± 2.50           |
| **DMOPSO**         | 23.30 ± 0.27          | 2.58 ± 2.19             | 38.12 ± 23.18          | 23.20 ± 0.27          | 2.24 ± 2.25             | 37.39 ± 20.79          |
| **PSO-normal**     | 23.29 ± 0.28          | 2.54 ± 2.01             | 67.54 ± 57.82          | 23.34 ± 0.27          | 3.43 ± 2.25             | 31.77 ± 16.63          |
| **RCH**            | 23.39 ± 0.27          | 3.38 ± 1.96             | 36.72 ± 0.55           | 23.25 ± 0.27          | 2.35 ± 1.96             | 35.27 ± 0.31           |


## 4.3 Cargo balancing/loading with Large Language Model optimization

| **Method**      | **B777 MAC(%)↓** | **B777 INDEX(%)↓** | **B777 TIME(s)↓** | **B787 MAC(%)↓** | **B787 INDEX(%)↓** | **B787 TIME(s)↓** |
|-----------------|------------------|--------------------|-------------------|------------------|--------------------|-------------------|
| **HGA**         | 23.44 ± 0.22      | 3.75 ± 1.65        | 4.91 ± 2.08       | 23.44 ± 0.21      | 3.73 ± 1.55        | 2.50 ± 1.12       |
| **GA-normal**   | 23.43 ± 0.23      | 3.66 ± 1.67        | 2.39 ± 0.76       | 23.46 ± 0.21      | 3.89 ± 1.58        | 1.29 ± 0.17       |
| **DMOPSO**      | 23.32 ± 0.28      | 3.21 ± 2.30        | 3.28 ± 2.23       | 23.39 ± 0.26      | 3.79 ± 2.10        | 1.60 ± 0.81       |
| **PSO-normal**  | 23.39 ± 0.28      | 3.79 ± 2.30        | 5.80 ± 4.09       | 23.39 ± 0.29      | 3.78 ± 2.37        | 1.19 ± 0.74       |
| **RCH**         | 23.39 ± 0.26      | 3.40 ± 1.91        | 3.66 ± 0.03       | 23.42 ± 0.24      | 3.61 ± 1.76        | 0.72 ± 0.01       |

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

# 7. Citation
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
