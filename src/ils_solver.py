import logging
import random
import time

import networkx as nx

# This is crucial for high beta values where analytical optimization is needed.
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

        # Pre-compute Dijkstra distances for O(1) access during the loop.
        # Essential for performance since the graph is sparse.
        self.shortest_dists = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path(problem.graph, weight='dist'))

        self.cities = [n for n in problem.graph.nodes if n != 0]

        # Adaptive Tuning:
        # If the landscape is rugged (Beta >= 1.5), we need a stronger 'kick'
        # (perturbation) to escape local optima.
        if problem.beta >= 1.5:
            self.perturbation_strength = 3
        else:
            self.perturbation_strength = 2

    def solve(self):
        start_global = time.time()

        # Initialization (Exploration)
        # Start with a random permutation of cities.
        current_solution = self._generate_initial_solution()

        # First Local Search (Exploitation)
        # Apply Hill Climbing to reach the first Local Optimum.
        current_solution = self._geometric_local_search(current_solution)

        # Evaluate the real cost using the Split Algorithm (decoding TSP tour to VRP trips)
        current_cost, current_logical_split = self._split_path(current_solution)
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
            # Make a non-local move to escape the current local optimum.
            perturbed_solution = self._perturb(current_solution)

            # Local Search (Hill Climbing)
            # Optimize the new candidate. Note: We optimize Geometric Distance
            # for speed, assuming spatial locality correlates with lower cost.
            refined_solution = self._geometric_local_search(perturbed_solution)

            # Evaluation
            # Use Prins' Split Algorithm to determine optimal depot returns.
            refined_cost_est, refined_logical = self._split_path(refined_solution)

            # Acceptance Criterion
            # If the new local optimum is better, we move there.
            if refined_cost_est < current_cost:
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
                # If we are stuck in a basin of attraction for too long, random restart.
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
                pass  # Fallback to estimated cost

        return best_physical_path, best_cost

    def _generate_initial_solution(self):
        """Random permutation for initialization."""
        sol = self.cities[:]
        random.shuffle(sol)
        return sol

    def _geometric_local_search(self, tour):
        """
        Hill Climbing (First Improvement) on Geometric Distance.

        We optimize the TSP path (pure distance) instead of the full weight function
        inside the loop to avoid O(N^2) heavy calculations at every step.
        """
        best_tour = tour[:]
        n = len(best_tour)
        improved = True
        dists = self.shortest_dists

        while improved:
            improved = False
            # Standard 2-Opt implementation
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Get nodes (handling wrap-around implies 0/Depot connection)
                    node_a = best_tour[i - 1] if i > 0 else 0
                    node_b = best_tour[i]
                    node_c = best_tour[j]
                    node_d = best_tour[j + 1] if j < n - 1 else 0

                    current_d = dists[node_a][node_b] + dists[node_c][node_d]
                    new_d = dists[node_a][node_c] + dists[node_b][node_d]

                    # First Improvement strategy
                    if new_d < current_d - 1e-6:
                        best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                        improved = True
                        break
                if improved: break
        return best_tour

    def _perturb(self, solution):
        """
        Double Bridge Move.
        A 'Kick' that disrupts the order more than a simple swap,
        helping to jump out of local optima while preserving some sub-structures.
        """
        new_sol = solution[:]
        n = len(new_sol)
        if n < 4: return new_sol

        # Split into 4 segments and reconnect A-D-C-B
        pos = sorted(random.sample(range(1, n), 3))
        p1, p2, p3 = pos
        return new_sol[:p1] + new_sol[p3:] + new_sol[p2:p3] + new_sol[p1:p2]

    def _split_path(self, tour):
        """
        Prins' Split Algorithm.
        Converts the 'Giant Tour' (TSP) into optimal trips (VRP).
        Builds a DAG where edges represent feasible trips and finds the shortest path.
        """
        n = len(tour)
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1)
        V[0] = 0

        alpha = self.problem.alpha
        beta = self.problem.beta
        dists = self.shortest_dists
        gold_map = nx.get_node_attributes(self.problem.graph, 'gold')

        # Optimization: Limit lookahead when Beta is high to speed up convergence.
        max_lookahead = n if beta < 1.5 else 5

        for i in range(n):
            if V[i] == float('inf'): continue

            load = 0.0
            cost = 0.0

            u = tour[i]
            cost += dists[0][u]
            load += gold_map[u]

            limit = min(n + 1, i + 1 + max_lookahead)

            for j in range(i + 1, limit):
                curr_node = tour[j - 1]

                if j > i + 1:
                    prev_node = tour[j - 2]
                    d = dists[prev_node][curr_node]

                    # HARD PRUNING:
                    # If Beta >= 2, moving with weight is exponentially expensive.
                    # If the move cost is > 2.5x the distance, it's better to return to depot.
                    # This heuristic prevents exploring infeasible/expensive branches.
                    if beta >= 2.0 and load > 0:
                        move_c = d + (alpha * d * load) ** beta
                        if move_c > 2.5 * d:
                            break

                    cost += d + (alpha * d * load) ** beta
                    load += gold_map[curr_node]

                d_home = dists[curr_node][0]
                return_c = d_home + (alpha * d_home * load) ** beta

                total = cost + return_c

                # Bellman equation update
                if V[i] + total < V[j]:
                    V[j] = V[i] + total
                    P[j] = i

        # Reconstruct the logical trips from Predecessor array P
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
        Feasibility check: The graph is sparse (Density < 1).
        We must ensure that going from A to B uses the actual shortest path
        if no direct edge exists.
        """
        physical = []
        physical.append(logical_path[0])
        dists = self.shortest_paths

        for k in range(len(logical_path) - 1):
            u, _ = logical_path[k]
            v, v_gold = logical_path[k + 1]
            if u == v: continue

            if self.problem.graph.has_edge(u, v):
                physical.append((v, v_gold))
            else:
                # Use cached Dijkstra path to fill the gap
                path = dists[u][v]
                for node in path[1:]:
                    # Only pick gold at the target city 'v', intermediate nodes are transit
                    g = v_gold if node == v else 0.0
                    physical.append((node, g))
        return physical