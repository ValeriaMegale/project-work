import logging
import random
import time

import networkx as nx
from src.utils import path_cost

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

    def __init__(self, problem, max_iterations=200, max_time=5):
        self.problem = problem
        self.max_iterations = max_iterations
        self.max_time = max_time

        self.shortest_paths_cache = {}
        self._dist_from_source = {}

        self.cities = [n for n in problem.graph.nodes if n != 0]

        # Pre-extract gold map once (used heavily in split)
        self.gold_map = nx.get_node_attributes(problem.graph, 'gold')

        # Adaptive Tuning:
        # If the landscape is rugged (Beta >= 1.5), we need a stronger 'kick'
        # (perturbation) to escape local optima.
        if problem.beta >= 1.5:
            self.perturbation_strength = 3
        else:
            self.perturbation_strength = 2

    def _get_distance(self, u, v):
        """Get shortest distance between u and v with caching.

        We run a single-source Dijkstra from each origin node at most once,
        then reuse its distance dict for all destinations.
        """
        if u not in self._dist_from_source:
            self._dist_from_source[u] = nx.single_source_dijkstra_path_length(
                self.problem.graph, u, weight="dist"
            )
        return self._dist_from_source[u][v]

        

    def solve(self):
        start_global = time.time()
        absolute_time_limit = 570  # 9 minutes 30 seconds hard limit
        
        # print(f"🔍 ILS: Starting with {len(self.cities)} cities, max_time={self.max_time}s")

        # Initialization (Exploration)
        # Start with a random permutation of cities.
        # print("🔍 ILS: Generating initial solution...")
        current_solution = self._generate_initial_solution()

        # First Local Search (Exploitation)
        # Apply Hill Climbing to reach the first Local Optimum.
        # print("🔍 ILS: Running initial local search...")
        current_solution = self._geometric_local_search(current_solution)
        # print(f"🔍 ILS: Initial local search completed in {time.time() - start_global:.2f}s")

        # Evaluate the real cost using the Split Algorithm (decoding TSP tour to VRP trips)
        # print("🔍 ILS: Running initial split...")
        current_cost, current_logical_split = self._split_path(current_solution)
        # print("🔍 ILS: Reconstructing physical path...")
        current_physical_path = self._reconstruct_physical_path(current_logical_split)
        # print(f"🔍 ILS: Initial setup completed in {time.time() - start_global:.2f}s")

        best_solution = current_solution[:]
        best_cost = current_cost
        best_physical_path = current_physical_path

        iter_no_improv = 0

        # Iterate (Tweak -> Local Search -> Accept)
        for i in range(self.max_iterations):
            elapsed_time = time.time() - start_global
            if elapsed_time > self.max_time or elapsed_time > absolute_time_limit:
                # print(f"🔍 ILS: Breaking due to timeout at iteration {i}, elapsed: {elapsed_time:.2f}s")
                break
            
            if i % 10 == 0:
                # print(f"🔍 ILS: Iteration {i}/{self.max_iterations}, elapsed: {elapsed_time:.2f}s, best_cost: {best_cost:.2f}")
                pass

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

        # print(f"🔍 ILS: Main loop completed after {i+1} iterations")
        
        # Apply path_optimizer only once at the end on best solution
        if path_optimizer and self.problem.beta > 1:
            # print("🔍 ILS: Applying path optimizer...")
            try:
                best_physical_path = path_optimizer(best_physical_path, self.problem)
                
                best_cost = path_cost(self.problem, best_physical_path)
                # print("🔍 ILS: Path optimizer completed")
            except Exception:
                # print("🔍 ILS: Path optimizer failed, using fallback")
                pass  # Fallback to estimated cost

        total_time = time.time() - start_global
        # print(f"🔍 ILS: Solve completed in {total_time:.2f}s, final cost: {best_cost:.2f}")
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
        n = len(tour)
        
        # For very large instances, skip local search entirely
        if n > 500:
            # print(f"🔍 ILS: Skipping local search for large instance (n={n})")
            return tour
            
        # print(f"🔍 ILS: Running local search on tour of size {n}")
        best_tour = tour[:]
        improved = True
        iterations = 0
        max_iterations = min(10, n // 10)  # Much more aggressive limits

        while improved and iterations < max_iterations:
            iterations += 1
            improved = False
            # For large instances, very limited sampling
            if n > 100:
                pairs_to_check = min(n, 100)  # At most 100 pairs for large instances
                pairs = []
                for _ in range(pairs_to_check):
                    i = random.randint(0, n-3)
                    j = random.randint(i+2, n-1)
                    pairs.append((i, j))
            else:
                # Standard 2-Opt implementation for small instances only
                pairs = [(i, j) for i in range(n-1) for j in range(i+1, n)]
                
            for i, j in pairs:
                # Get nodes (handling wrap-around implies 0/Depot connection)
                node_a = best_tour[i - 1] if i > 0 else 0
                node_b = best_tour[i]
                node_c = best_tour[j]
                node_d = best_tour[j + 1] if j < n - 1 else 0

                current_d = self._get_distance(node_a, node_b) + self._get_distance(node_c, node_d)
                new_d = self._get_distance(node_a, node_c) + self._get_distance(node_b, node_d)

                # First Improvement strategy
                if new_d < current_d - 1e-6:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    improved = True
                    break
        # print(f"🔍 ILS: Local search completed with {iterations} iterations")
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
        Converte il "Giant Tour" (TSP) in viaggi ottimi (VRP) usando un DP.
        """
        n = len(tour)
        # print(f"🔍 ILS: Running split algorithm on tour of size {n}")
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1)
        V[0] = 0.0

        alpha = self.problem.alpha
        beta = self.problem.beta

        # Local references per efficienza
        gold_map = self.gold_map
        get_dist = self._get_distance

        # Lookahead aggressivo per istanze grandi
        if n > 500:
            max_lookahead = 3
        elif n > 200:
            max_lookahead = 5
        elif beta >= 1.5:
            max_lookahead = min(10, n // 20)
        else:
            max_lookahead = min(20, n // 10)

        # print(f"🔍 ILS: Split using max_lookahead={max_lookahead} for n={n}, beta={beta:.2f}")

        beta_ge_2 = beta >= 2.0

        for i in range(n):
            if V[i] == float('inf'):
                continue

            load = 0.0
            cost = 0.0

            # Primo nodo del viaggio potenziale: deposito -> tour[i]
            u = tour[i]
            cost += get_dist(0, u)
            load += gold_map[u]

            limit = min(n + 1, i + 1 + max_lookahead)

            for j in range(i + 1, limit):
                curr_node = tour[j - 1]

                if j > i + 1:
                    prev_node = tour[j - 2]
                    d = get_dist(prev_node, curr_node)

                    # HARD PRUNING come prima
                    if beta_ge_2 and load > 0:
                        move_c = d + (alpha * d * load) ** beta
                        if move_c > 2.5 * d:
                            break

                    cost += d + (alpha * d * load) ** beta
                    load += gold_map[curr_node]

                d_home = get_dist(curr_node, 0)
                return_c = d_home + (alpha * d_home * load) ** beta

                total = cost + return_c

                if V[i] + total < V[j]:
                    V[j] = V[i] + total
                    P[j] = i

        # Ricostruzione dei viaggi logici da P
        # print(f"🔍 ILS: Reconstructing logical trips from split result")
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
                full_logical.append((node, float(gold_map[node])))
        full_logical.append((0, 0.0))

        # print(f"🔍 ILS: Split algorithm completed, found {len(trips)} trips")
        return V[n], full_logical

    def _reconstruct_physical_path(self, logical_path):
        """
        Feasibility check: The graph is sparse (Density < 1).
        We must ensure that going from A to B uses the actual shortest path
        if no direct edge exists.
        """
        # print(f"🔍 ILS: Reconstructing physical path from {len(logical_path)} logical steps")
        physical = []
        physical.append(logical_path[0])

        for k in range(len(logical_path) - 1):
            u, _ = logical_path[k]
            v, v_gold = logical_path[k + 1]
            if u == v: continue

            if self.problem.graph.has_edge(u, v):
                physical.append((v, v_gold))
            else:
                # Calculate path on-demand and cache it
                path_key = (u, v)
                if path_key not in self.shortest_paths_cache:
                    self.shortest_paths_cache[path_key] = nx.shortest_path(self.problem.graph, u, v, weight='dist')
                
                path = self.shortest_paths_cache[path_key]
                for node in path[1:]:
                    # Only pick gold at the target city 'v', intermediate nodes are transit
                    g = v_gold if node == v else 0.0
                    physical.append((node, g))
        # print(f"🔍 ILS: Physical path reconstruction completed, {len(physical)} total steps")
        return physical
