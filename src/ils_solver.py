import logging
import random
import time
import numpy as np
import networkx as nx
import scipy.sparse
from scipy.sparse.csgraph import shortest_path
from numba import jit


try:
    from .beta_optimizer import path_optimizer
except ImportError:
    try:
        from beta_optimizer import path_optimizer
    except ImportError:
        path_optimizer = None


@jit(nopython=True, cache=True)
def fast_2opt(tour, dist_matrix):
    best_tour = tour.copy()
    n = len(best_tour)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                node_a = best_tour[i - 1] if i > 0 else 0
                node_b = best_tour[i]
                node_c = best_tour[j]
                node_d = best_tour[j + 1] if j < n - 1 else 0

                current_d = dist_matrix[node_a, node_b] + dist_matrix[node_c, node_d]
                new_d = dist_matrix[node_a, node_c] + dist_matrix[node_b, node_d]

                if new_d < current_d - 1e-6:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best_tour
    #
    # def fast_2opt(tour, dist_matrix):
    #     best_tour = tour.copy()
    #     n = len(best_tour)
    #     improved = True
    #     while improved:
    #         improved = False
    #         for i in range(n - 1):
    #             for j in range(i + 1, n):
    #                 node_a = best_tour[i - 1] if i > 0 else 0
    #                 node_b = best_tour[i]
    #                 node_c = best_tour[j]
    #                 node_d = best_tour[j + 1] if j < n - 1 else 0
    #
    #                 current_d = dist_matrix[node_a, node_b] + dist_matrix[node_c, node_d]
    #                 new_d = dist_matrix[node_a, node_c] + dist_matrix[node_b, node_d]
    #
    #                 if new_d < current_d - 1e-6:
    #                     best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
    #                     improved = True
    #                     break
    #             if improved: break
    #     return best_tour
    #

class IteratedLocalSearchSolver:
    def __init__(self, problem, max_iterations=200, max_time=25):
        self.problem = problem
        self.max_iterations = max_iterations
        self.max_time = max_time

        # Convert graph to sparse matrix for fast Dijkstra
        adj_matrix = nx.to_scipy_sparse_array(problem.graph, weight='dist', format='csr')

        self.dist_matrix, self.predecessors = shortest_path(
            csgraph=adj_matrix,
            directed=problem.graph.is_directed(),
            return_predecessors=True
        )

        self.cities = np.array([n for n in problem.graph.nodes if n != 0], dtype=np.int32)
        self.gold_map = nx.get_node_attributes(problem.graph, 'gold')

        if problem.beta >= 1.5:
            self.perturbation_strength = 3
        else:
            self.perturbation_strength = 2

    def solve(self):
        start_global = time.time()

        current_solution = self._generate_initial_solution()
        current_solution = fast_2opt(current_solution, self.dist_matrix)
        current_cost, current_logical_split = self._split_path(current_solution)

        current_physical_path = self._reconstruct_physical_path(current_logical_split)

        if path_optimizer and self.problem.beta > 1:
            try:
                current_physical_path = path_optimizer(current_physical_path, self.problem)
                current_cost = self.problem.path_cost(current_physical_path)
            except Exception:
                pass

        best_solution = current_solution.copy()
        best_cost = current_cost
        best_physical_path = current_physical_path

        iter_no_improv = 0

        for i in range(self.max_iterations):
            if time.time() - start_global > self.max_time:
                break

            perturbed_solution = self._perturb(current_solution)
            refined_solution = fast_2opt(perturbed_solution, self.dist_matrix)
            refined_cost_est, refined_logical = self._split_path(refined_solution)

            if refined_cost_est < current_cost:
                refined_physical = self._reconstruct_physical_path(refined_logical)

                if path_optimizer and self.problem.beta > 1:
                    try:
                        refined_physical = path_optimizer(refined_physical, self.problem)
                        real_cost = self.problem.path_cost(refined_physical)
                    except:
                        real_cost = refined_cost_est
                else:
                    real_cost = refined_cost_est

                current_solution = refined_solution
                current_cost = real_cost
                iter_no_improv = 0

                if real_cost < best_cost:
                    best_cost = real_cost
                    best_solution = current_solution
                    best_physical_path = refined_physical
            else:
                iter_no_improv += 1

                if iter_no_improv > 35:
                    current_solution = self._generate_initial_solution()
                    current_solution = fast_2opt(current_solution, self.dist_matrix)
                    c, l = self._split_path(current_solution)
                    p = self._reconstruct_physical_path(l)

                    if path_optimizer and self.problem.beta > 1:
                        p = path_optimizer(p, self.problem)
                        c = self.problem.path_cost(p)

                    current_cost = c
                    iter_no_improv = 0

        return best_physical_path, best_cost

    def _generate_initial_solution(self):
        sol = self.cities.copy()
        np.random.shuffle(sol)
        return sol

    def _perturb(self, solution):
        new_sol = solution.copy()
        n = len(new_sol)
        if n < 4: return new_sol

        pos = sorted(random.sample(range(1, n), 3))
        p1, p2, p3 = pos

        return np.concatenate((
            new_sol[:p1],
            new_sol[p3:],
            new_sol[p2:p3],
            new_sol[p1:p2]
        ))

    def _split_path(self, tour):
        n = len(tour)
        V = np.full(n + 1, np.inf)
        P = np.zeros(n + 1, dtype=int)
        V[0] = 0

        alpha = self.problem.alpha
        beta = self.problem.beta
        dists = self.dist_matrix
        gold = self.gold_map

        max_lookahead = n if beta < 1.5 else 5

        for i in range(n):
            if np.isinf(V[i]): continue

            load = 0.0
            cost = 0.0

            u = tour[i]
            cost += dists[0, u]
            load += gold[u]

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
                    load += gold[curr_node]

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
                full_logical.append((node, gold[node]))
        full_logical.append((0, 0.0))

        return V[n], full_logical

    def _reconstruct_physical_path(self, logical_path):
        physical = []
        physical.append(logical_path[0])

        preds = self.predecessors

        for k in range(len(logical_path) - 1):
            u, _ = logical_path[k]
            v, v_gold = logical_path[k + 1]
            if u == v: continue

            path_segment = []
            curr = v

            # Backtrack using predecessor matrix
            while curr != u:
                path_segment.append(curr)
                curr = preds[u, curr]
                if curr == -9999:  # Should not happen in a connected component
                    break

            path_segment.reverse()

            for node in path_segment:
                g = v_gold if node == v else 0.0
                physical.append((node, g))

        return physical