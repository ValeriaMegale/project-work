# Gold Collector Problem Solver

This project provides a solution to the gold collector problem, a variation of the Vehicle Routing Problem (VRP), using a sophisticated heuristic merge strategy. The goal is to find the minimum cost path to collect gold from various cities and return it to a central depot, considering that the travel cost increases with the weight of the gold carried.

## Core Strategy: Merge Optimizer

The project now exclusively uses a single, powerful optimization strategy: the **Merge Optimizer**. This approach has proven to be highly effective and has been retained after a significant refactoring to simplify the codebase.

### Algorithm Overview

The Merge Optimizer is a constructive heuristic that builds an efficient solution by iteratively merging and reordering trips. It operates as follows:

1.  **Initialization**: The algorithm starts with a set of candidate nodes (cities with gold).
2.  **Heuristic Construction**: It incrementally builds a path by selecting the next city to visit based on a heuristic that balances travel distance and the additional cost incurred by carrying more gold. This selection is not purely greedy; it includes a degree of randomness to explore a wider range of solutions and avoid getting stuck in local optima.
3.  **Path Refinement**: The algorithm uses a k-nearest neighbors approach to consider a small set of promising next steps at each stage, rather than evaluating all possibilities. This makes the process highly efficient.
4.  **Dynamic Weight Management**: The cost function `cost = dist + (dist * α * weight)^β` is central to the decision-making process. The algorithm intelligently decides when it is more cost-effective to travel directly to the next city versus returning to the depot to unload gold.

### Key Features

*   **Adaptive Heuristic**: The core of the optimizer is an adaptive incremental heuristic with controlled noise, which helps to avoid local minima.
*   **Efficient Path Finding**: It uses Dijkstra's algorithm to pre-calculate distances, making the subsequent heuristic calculations very fast.
*   **Scalability**: The use of a k-neighbors approach and other optimizations allows the solver to handle large problem instances efficiently.
*   **Focused and Maintained**: By concentrating on a single, effective strategy, the codebase is cleaner, more maintainable, and easier to understand.

## How to Run

The main entry point for the solver is the `problem_solver` function in `src/solver_framework.py`. This function takes a `Problem` instance and returns the optimized path and its total cost.

```python
from Problem import Problem
from src.solver_framework import problem_solver

# Create a problem instance
problem = Problem(...)

# Solve the problem
path, cost = problem_solver(problem)

print(f"Optimal path: {path}")
print(f"Total cost: {cost}")

```
This project is the result of a refactoring effort to distill the most effective solution from a series of experiments with different approaches. The Merge Optimizer stands as the most robust and efficient strategy developed.
