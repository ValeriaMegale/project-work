import logging
import random
import time
import numpy as np
import networkx as nx
import scipy.sparse
from scipy.sparse.csgraph import shortest_path

try:
    from .beta_optimizer import path_optimizer
except ImportError:
    try:
        from beta_optimizer import path_optimizer
    except ImportError:
        path_optimizer = None
        logging.warning("Beta Optimizer not found. Performance on high Beta will suffer.")


class IteratedLocalSearchSolver:
    """
    Implementation of Iterated Local Search (ILS).

    Strategy:
    1. Reach a Local Optimum using Hill Climbing (Local Search).
    2. Apply a 'Tweak' (Perturbation) to jump out of the basin of attraction.
    3. Restart Local Search from the new point.
    """

    def __init__(self, problem, max_iterations=200, max_time=25):
        self.problem = problem
        self.max_iterations = max_iterations
        self.max_time = max_time

        adj_matrix = nx.to_scipy_sparse_array(problem.graph, weight='dist', format='csr')

        self.dist_matrix, self.predecessors = shortest_path(
            csgraph=adj_matrix,
            directed=problem.graph.is_directed(),
            return_predecessors=True
        )

        self.cities = [n for n in problem.graph.nodes if n != 0]

        # Adaptive Tuning:
        if problem.beta >= 1.5:
            self.perturbation_strength = 3
        else:
            self.perturbation_strength = 2

    def solve(self):
        start_global = time.time()

        # Initialization (Exploration)
        current_solution = self._generate_initial_solution()

        # First Local Search (Exploitation)
        current_solution = self._geometric_local_search(current_solution)

        # Evaluate the real cost using the Split Algorithm
        current_cost, current_logical_split = self._split_path(current_solution)

        # Reconstruct physical path only when needed (saves time inside the loop)
        current_physical_path = self._reconstruct_physical_path(current_logical_split)

        best_solution = current_solution[:]
        best_cost = current_cost
        best_physical_path = current_physical_path

        iter_no_improv = 0

        # Iterate (Tweak -> Local Search -> Accept)
        for i in range(self.max_iterations):
            if time.time() - start_global > self.max_time:
                break

            # Perturbation (The "Tweak")
            perturbed_solution = self._perturb(current_solution)

            # Local Search (Hill Climbing)
            refined_solution = self._geometric_local_search(perturbed_solution)

            # Evaluation
            refined_cost_est, refined_logical = self._split_path(refined_solution)

            # Acceptance Criterion
            if refined_cost_est < current_cost:

                # Reconstruct full path only on acceptance
                refined_physical = self._reconstruct_physical_path(refined_logical)

                # Update current state
                current_solution = refined_solution
                current_cost = refined_cost_est
                iter_no_improv = 0

                # Update Global Best
                if refined_cost_est < best_cost:
                    best_cost = refined_cost_est
                    best_solution = current_solution
                    best_physical_path = refined_physical
            else:
                iter_no_improv += 1
                # Restart Strategy
                if iter_no_improv > 35:
                    current_solution = self._generate_initial_solution()
                    current_solution = self._geometric_local_search(current_solution)
                    current_cost, l = self._split_path(current_solution)
                    iter_no_improv = 0

        # Apply path_optimizer only once at the end on best solution
        if path_optimizer and self.problem.beta > 1:
            try:
                best_physical_path = path_optimizer(best_physical_path, self.problem)
                best_cost = self.problem.path_cost(best_physical_path)
            except Exception:
                pass

        return best_physical_path, best_cost

    def _generate_initial_solution(self):
        """Random permutation for initialization."""
        sol = self.cities[:]
        random.shuffle(sol)
        return sol

    def _geometric_local_search(self, tour):
        """
        Hill Climbing (First Improvement) on Geometric Distance.
        Uses Numpy matrix access which is faster than dictionary lookup.
        """
        best_tour = tour[:]
        n = len(best_tour)
        improved = True

        # Direct access to numpy array for speed
        dists = self.dist_matrix

        while improved:
            improved = False
            # Standard 2-Opt implementation
            for i in range(n - 1):
                for j in range(i + 1, n):
                    node_a = best_tour[i - 1] if i > 0 else 0
                    node_b = best_tour[i]
                    node_c = best_tour[j]
                    node_d = best_tour[j + 1] if j < n - 1 else 0

                    # Access numpy array [row, col]
                    current_d = dists[node_a, node_b] + dists[node_c, node_d]
                    new_d = dists[node_a, node_c] + dists[node_b, node_d]

                    # First Improvement strategy
                    if new_d < current_d - 1e-6:
                        best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                        improved = True
                        break
                if improved: break
        return best_tour

    def _perturb(self, solution):
        """Double Bridge Move."""
        new_sol = solution[:]
        n = len(new_sol)
        if n < 4: return new_sol

        pos = sorted(random.sample(range(1, n), 3))
        p1, p2, p3 = pos
        return new_sol[:p1] + new_sol[p3:] + new_sol[p2:p3] + new_sol[p1:p2]

    def _split_path(self, tour):
        """
        Prins' Split Algorithm.
        Optimized to use Scipy distance matrix.
        """
        n = len(tour)
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1)
        V[0] = 0

        alpha = self.problem.alpha
        beta = self.problem.beta

        # Use numpy array for distances
        dists = self.dist_matrix
        gold_map = nx.get_node_attributes(self.problem.graph, 'gold')

        max_lookahead = n if beta < 1.5 else 5

        for i in range(n):
            if V[i] == float('inf'): continue

            load = 0.0
            cost = 0.0

            u = tour[i]
            cost += dists[0, u]  # Numpy access
            load += gold_map[u]

            limit = min(n + 1, i + 1 + max_lookahead)

            for j in range(i + 1, limit):
                curr_node = tour[j - 1]

                if j > i + 1:
                    prev_node = tour[j - 2]
                    d = dists[prev_node, curr_node]

                    if beta >= 2.0 and load > 0:
                        move_c = d + (alpha * d * load) ** beta
                        if move_c > 2.5 * d:
                            break

                    cost += d + (alpha * d * load) ** beta
                    load += gold_map[curr_node]

                d_home = dists[curr_node, 0]
                return_c = d_home + (alpha * d_home * load) ** beta

                total = cost + return_c

                if V[i] + total < V[j]:
                    V[j] = V[i] + total
                    P[j] = i

        curr = n
        trips = []
        while curr > 0:
            prev = P[curr]
            trips.append(tour[prev:curr])
            curr = prev
        trips.reverse()

        full_logical = []
        for trip in trips:
            full_logical.append((0, 0.0))
            for node in trip:
                full_logical.append((node, gold_map[node]))
        full_logical.append((0, 0.0))

        return V[n], full_logical

    def _reconstruct_physical_path(self, logical_path):
        """
        Reconstructs the physical path using the Scipy predecessor matrix.
        Instead of looking up cached paths (RAM heavy), we rebuild them on-the-fly (CPU fast with C++ backend).
        """
        physical = []
        physical.append(logical_path[0])

        # Predecessors matrix from scipy.sparse.csgraph.shortest_path
        preds = self.predecessors

        for k in range(len(logical_path) - 1):
            u, _ = logical_path[k]
            v, v_gold = logical_path[k + 1]
            if u == v: continue

            # If there is a direct edge in the original graph, check if it matches the shortest distance
            # NOTE: Scipy preds matrix handles the shortest path logic automatically.

            # Reconstruct path from u to v using predecessors
            path_segment = []
            curr = v

            # Backtrack from target (v) to source (u)
            while curr != u:
                path_segment.append(curr)
                prev = preds[u, curr]

                # If prev is -9999 (default no path), we break to avoid infinite loop,
                # though in a connected component this shouldn't happen.
                if prev == -9999:
                    break
                curr = prev

            # The segment was built backwards (v -> ... -> node after u), so reverse it
            path_segment.reverse()

            for node in path_segment:
                g = v_gold if node == v else 0.0
                physical.append((node, g))

        return physical