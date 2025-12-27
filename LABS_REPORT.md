# Labs Report

This document summarizes the algorithmic strategies implemented for the three course laboratories, detailing the initial solutions and the subsequent optimizations derived from peer reviews and performance analysis on complex instances.

---

## Lab 1: Set Covering / Knapsack Problem

The first laboratory required solving constrained optimization problems (variants of the Knapsack Problem) using local search algorithms and evolutionary algorithms.

### Implemented Solution
The solution is based on a hybrid approach combining genetic algorithms with local search techniques (Hill Climbing and Simulated Annealing).

* **Representation and Validation:**
    * The solution is represented as a boolean matrix where `solution[k, i] = True` means item `i` is in knapsack `k`.
    * Functions `is_valid` and `validate` are used to ensure no item is duplicated and weight constraints are not violated.
    * The `evaluate` function calculates the total value of items; if the solution is invalid, it initially returned a penalizing value (-1.0).

* **Hill Climber & Simulated Annealing:**
    * A standard **Hill Climber** was implemented that explores the neighborhood via the `tweak` function (swapping items or adding/removing).
    * A **Fast Simulated Annealing** version was developed and used as a local search engine within the genetic algorithm.

* **Evolutionary Algorithm (Memetic):**
    * The main algorithm (`my_algorithm`) is a **Memetic Algorithm** combining evolution and individual learning.
    * **Initialization:** Generates a population of random valid solutions (`create_random_valid_solution`).
    * **Crossover:** Offspring are created by mixing the item assignments of parents.
    * **Local Refinement:** Each offspring undergoes an improvement process via *Simulated Annealing* (memetic phase) before evaluation, allowing it to escape local optima and "learn".

### Post-Review Modifications
Following the review, critical issues were identified regarding search gradient management and solution space navigation.

1.  **Solving the "Fast Hill Climber Problem" (Stagnation):**
    * **Critique:** Using strict inequalities (`>`) prevented the algorithm from traversing plateaus (neutral moves), causing premature blocking in local optima.
    * **Modification:** The `hill_climber_fast` version was discarded. The logic was refined to accept moves with equal cost (`>=`), allowing navigation across flat spaces. Additionally, a stochastic component was introduced to accept worse solutions with a decaying probability, similar to Simulated Annealing, to escape local minima.

2.  **Handling the "Sea of Invalid Solutions":**
    * **Critique:** Returning a flat score (-1.0) for invalid solutions eliminated gradient information. In constrained spaces, random moves often led to invalid states indistinguishable from one another, turning the search into a "random walk".
    * **Modification - Start from Void:** The algorithm now starts from a `void_solution` (all zeros), which is guaranteed to be valid (weight 0 ≤ capacity).
    * **Modification - Immediate Validation:** Instead of penalizing retrospectively, the `hill_climber` immediately checks validity via `validate()`. Moves leading to invalid states are discarded or handled by the search logic, maintaining guidance toward promising regions.

---

## Lab 2: Genetic Algorithm for TSP

The second laboratory addressed the Traveling Salesperson Problem (TSP) using a Genetic Algorithm.

### Implemented Solution
The base structure of the algorithm follows the standard for permutation problems.

* **Representation:**
    * A solution is an array of integers representing a cyclic permutation of cities $[0, \dots, N-1]$.
    * Cost is calculated as the sum of edge weights in the tour (including the return to the start).

* **Genetic Operators:**
    * **Selection:** *Tournament Selection* with parameter $k$. Low $k$ favors diversity, high $k$ increases selective pressure.
    * **Crossover:** *Order Crossover (OX1)*. Selects two cut points, copies the segment from the first parent, and fills the rest preserving the relative order of the second parent. This guarantees permutation validity.
    * **Initial Mutation:** *Swap Mutation*. Simply swapped two random cities with probability `mutation_rate`.

* **Evolutionary Cycle:**
    * Random initialization, evaluation, elitism (copying the best), reproduction, and generational replacement.

### Post-Review Modifications (Optimization for 1000 Cities)
To address the 1000-city instance, the original solution was inefficient. The following structural changes were made based on the review.

1.  **New Mutation Operator (Inversion Mutation):**
    * **Issue:** *Swap Mutation* was too destructive to adjacency information (breaking too many valid edges).
    * **Modification:** **Inversion Mutation** was introduced, which reverses a random subsequence of the tour. This acts like a random **2-opt** move, helping to "untangle crossing paths" while preserving much of the adjacency.

2.  **Hybrid Initialization:**
    * **Issue:** Purely random initialization on 1000 cities started from extremely low fitness, drastically slowing convergence.
    * **Modification:** Implemented **Hybrid Initialization**. A percentage of the population (e.g., 30%) is generated using a randomized **Greedy (Nearest Neighbor)** heuristic, while the rest remains random to maintain genetic diversity.

3.  **Memetic Algorithm (Local Search):**
    * **Modification:** Introduced a **Local Search (2-opt)** phase to refine solutions. It is applied systematically to the best individual (elitism) and with low probability to offspring, transforming the algorithm into a Memetic one.

---

## Lab 3: Pathfinding Algorithms

The third laboratory focused on graph search algorithms and comparative benchmarking.

### Implemented Algorithms

1.  **Greedy Best-First Search (`best_first_solver`):**
    * **Type:** Informed search algorithm.
    * **Logic:** Selects the next node based exclusively on a **heuristic function** (Euclidean distance to the target).
    * **Characteristics:** Prioritizes speed and efficiency. However, it is **non-optimal** and does not guarantee the shortest path, as it ignores the accumulated cost to reach the current node.

2.  **SPFA - Shortest Path Faster Algorithm (`solver`):**
    * **Type:** Optimal search algorithm (Optimization of Bellman-Ford).
    * **Logic:** Uses a queue to selectively relax only vertices whose distances have changed, avoiding the redundant calculations of standard Bellman-Ford.
    * **Characteristics:**
        * Handles graphs with **negative weights** and correctly detects **negative cycles**.
        * Guarantees the **optimal path**.
        * Significantly faster than Bellman-Ford on sparse graphs.

### Post-Review Modifications (Benchmarking)
For this laboratory, no algorithmic modifications to the solvers were required. The main request from the review was the implementation of a robust benchmark system.

* **Benchmark Function:** The `benchmark_algorithms` function was added in the notebook `problem-creator.ipynb`. This function statistically compares:
    * NetworkX Bellman-Ford (Ground Truth/Baseline).
    * Custom SPFA Solver.
    * Greedy Best-First Search.
* The benchmark measures and reports execution times, path costs found, and the "Optimality Gap" of the Greedy approach compared to the exact solution.