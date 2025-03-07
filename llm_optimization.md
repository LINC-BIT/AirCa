# LLM optimization

Heuristic search
algorithms achieve better performance but also consume much
longer time than combinatorial optimization algorithms. This is
because existing heuristic algorithms are designed for simplified
scenarios with small numbers of cargo loading constraints. For such
an issue, we leverage LLM to optimize the optimization process of
heuristic search algorithms

# Required Python Packages

To install all required packages, run the following pip commands:
```bash
pip install openai
```

## Example 1: GA optimization

Run the following command in Python:
```bash
python HGA_llm.py
```
![image](https://github.com/user-attachments/assets/1f913398-de23-416d-b066-ce6e5ead40f2)

## Example 2: PSO optimization

Run the following command in Python:
```bash
python DMOPSO_llm.py
```
![image](https://github.com/user-attachments/assets/ac50cc1a-7c41-471d-abab-610057674741)

