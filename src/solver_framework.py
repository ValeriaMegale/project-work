"""Solver entry point.

Drives the `goldcollector` genetic-algorithm solver and applies the beta
optimizer (`src/beta_optimizer.py`) for super-linear penalties. Exposes the
`problem_solver(problem) -> (path, cost)` contract that `s343540.py` relies on.
"""

import logging
from time import time

import networkx as nx

from Problem import Problem
from src.goldcollector import GAConfig, GeneticSolver, Instance
from src.goldcollector.postprocess import path_cost as walk_cost


def _config_for(n_nodes: int) -> GAConfig:
    """Fixed population / generations regardless of instance size."""
    return GAConfig(pop_size=100, generations=1000)


def _baseline_walk(problem: Problem) -> list[tuple[int, float]]:
    """The per-city round-trip baseline as an explicit feasible path.

    Its `path_cost` equals `problem.baseline()`; used as a safety net so the
    solver is never worse than the trivial baseline (notably for beta = 1, where
    the penalty is linear and the baseline is already near optimal).
    """
    graph = problem.graph
    gold = nx.get_node_attributes(graph, "gold")
    paths = nx.single_source_dijkstra_path(graph, source=0, weight="dist")
    walk = [(0, 0.0)]
    for city in graph.nodes:
        if city == 0:
            continue
        out = paths[city]
        for node in out[1:-1]:
            walk.append((node, 0.0))
        walk.append((city, gold[city]))          # collect at the city
        for node in reversed(out[:-1]):           # return to the depot
            walk.append((node, 0.0))
    return walk


def problem_solver(problem: Problem, optimize: bool = True) -> tuple[list[tuple[int, float]], float]:
    """Solve `problem` and return (path, cost).

    `path` is a list of (city, gold_picked_up) steps; the beta optimizer is
    applied automatically when beta > 1 (unless `optimize=False`). `cost` is
    computed internally (the edge-summed `postprocess.path_cost`), so the solver
    works with the original `Problem` API and does not depend on the project's
    added `Problem.path_cost`. Falls back to the baseline path if the GA fails
    to beat it.
    """
    start = time()
    instance = Instance.from_problem(problem)
    solution = GeneticSolver(
        instance, _config_for(problem.graph.number_of_nodes()), optimize=optimize
    ).solve()

    path = solution.walk
    cost = walk_cost(path, instance)

    baseline_path = _baseline_walk(problem)
    baseline_cost = walk_cost(baseline_path, instance)
    if baseline_cost < cost:
        logging.info(f"goldcollector: GA {cost:.2f} worse than baseline {baseline_cost:.2f}; using baseline")
        path, cost = baseline_path, baseline_cost

    logging.info(
        f"goldcollector: cost {cost:.2f} | steps {len(path)} | {time() - start:.2f}s"
    )
    return path, cost
