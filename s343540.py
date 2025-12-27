import logging
import time
from Problem import Problem
from src.solver_framework import problem_solver

def solution(problem: Problem) -> list[tuple[int, float]]:
    """
    Solve the problem using multiple solvers and select the best solution.
    """

    path, cost = problem_solver(problem)
    if path[0][0] == 0:
        path = path[1:]
    return path


def compare(problem: Problem) -> tuple[float, float, float]:
    """
    Compare the solution cost with the baseline cost.
    
    Returns:
        tuple: (improvement %, solution_cost, baseline_cost)
    """
    baseline_cost = problem.baseline()
    _, solution_cost = problem_solver(problem)
    improvement = (baseline_cost - solution_cost) / baseline_cost * 100
    return (improvement, solution_cost, baseline_cost)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = open("results.txt", "w")
    # Possible values: num_cities: 100, 1_000; density: 0.2, 1; alpha: 1, 2; beta: 1, 2
    for num_cities in [100]:
        for density in [0.2, 1]:
            for beta in [0.5, 1, 2]:
                for alpha in [1]:
                    print(f"Running Problem with {num_cities} cities, density={density}, alpha={alpha}, beta={beta}")
                    start_time = time.time()
                    problem = Problem(100, density=density, alpha=alpha, beta=beta, seed=42)
                    improvement, sol_cost, base_cost = compare(problem)
                    elapsed_time = time.time() - start_time
                    out.write(f"Density: {density}, Alpha: {alpha}, Beta: {beta} => Improvement: {improvement:.2f}%, Solution Cost: {sol_cost:.2f}, Baseline Cost: {base_cost:.2f}, Time: {elapsed_time:.2f}s\n")
    out.close()
