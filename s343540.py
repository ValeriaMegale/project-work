import logging
import multiprocessing
import time
from Problem import Problem
from src.merge_optimizer import merge_solver

def solution(problem: Problem) -> list[tuple[int, float]]:
    """
    Risolve il problema ed estrae il percorso ottimale completo.
    """
    multiprocessing.freeze_support()
    path, cost = merge_solver(problem)
    return path  # <- Mantiene il percorso intatto partendo da (0,0)


def compare(problem: Problem) -> tuple[float, float, float]:
    """
    Confronta il costo della soluzione euristica con il baseline di Dijkstra.
    """
    baseline_cost = problem.baseline()
    _, solution_cost = merge_solver(problem)
    improvement = (baseline_cost - solution_cost) / baseline_cost * 100
    return (improvement, solution_cost, baseline_cost)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)
    out = open("results.txt", "w")

    for num_cities in [100, 1000]:
        for density in [0.2, 1]:
            for beta in [1, 2]:
                for alpha in [1, 2]:
                    print(f"Running Problem with {num_cities} cities, density={density}, alpha={alpha}, beta={beta}")
                    start_time = time.time()
                    problem = Problem(num_cities=num_cities, density=density, alpha=alpha, beta=beta, seed=42)
                    improvement, sol_cost, base_cost = compare(problem)
                    elapsed_time = time.time() - start_time
                    out.write(
                        f"Density: {density}, Alpha: {alpha}, Beta: {beta} => Improvement: {improvement:.2f}%, Solution Cost: {sol_cost:.2f}, Baseline Cost: {base_cost:.2f}, Time: {elapsed_time:.2f}s\n")
    out.close()