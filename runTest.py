# src/runTest.py - test e confronto soluzioni per Gold Thief

import sys
from pathlib import Path
import pandas as pd
import networkx as nx

# Aggiunge la directory principale (parent di src) al path 
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Problem import Problem
from s343540 import solution



def compute_cost(problem, path):
    """
    Calcola il costo totale di un percorso secondo il modello del prof
    """
    alpha, beta = problem.alpha, problem.beta
    graph = problem.graph

    total_cost = 0.0
    carried_gold = 0.0

    for i in range(len(path) - 1):
        current_city, gold_loaded = path[i]
        next_city = path[i + 1][0]

        carried_gold += gold_loaded

        try:
            sp = nx.shortest_path(graph, current_city, next_city, weight="dist")
            d = nx.path_weight(graph, sp, weight="dist")
        except nx.NetworkXNoPath:
            return float("inf")

        total_cost += d + (d * alpha * carried_gold) ** beta

        if next_city == 0:
            carried_gold = 0.0

    return total_cost


def baseline_solution(problem):
    """
    Baseline semplice: visita ogni città singolarmente (0 -> città -> 0)
    """
    graph = problem.graph
    path = []

    for city in graph.nodes:
        if city == 0:
            continue
        gold = graph.nodes[city]["gold"]
        path.extend([(0, 0), (city, gold), (0, 0)])

    return path


def run_tests(verbose=True):
    # Test combinatori (piccolo-medio-grande) 
    sizes = [10, 50, 100, 200]  # Aggiunto 200
    densities = [0.2, 0.5, 1.0]
    alpha_values = [0.0, 1.0, 2.0, 4.0]
    beta_values = [0.5, 1.0, 2.0, 4.0]
    seed = 42

    results = []

    for size in sizes:
        for density in densities:
            for alpha in alpha_values:
                for beta in beta_values:

                    problem = Problem(
                        size,
                        density=density,
                        alpha=alpha,
                        beta=beta,
                        seed=seed
                    )

                    base_cost = compute_cost(problem, baseline_solution(problem))
                    my_cost = compute_cost(problem, solution(problem))
                    improvement = 100 * (base_cost - my_cost) / base_cost

                    results.append({
                        "Cities": size,
                        "Density": density,
                        "Alpha": alpha,
                        "Beta": beta,
                        "Baseline": round(base_cost, 2),
                        "goldThief": round(my_cost, 2),
                        "Improvement_%": round(improvement, 2)
                    })

                    if verbose:
                        print(f"\nTest: n={size}, density={density}, alpha={alpha}, beta={beta}")
                        print(f"  Baseline: {base_cost:.2f}")
                        print(f"  goldThief: {my_cost:.2f}")
                        print(f"  Improvement: {improvement:.2f}%")

    # Tabella combinatoria 
    df_results = pd.DataFrame(results)

    # Salva risultati in Excel 
    output_path = Path(__file__).parent / "results.xlsx"
    df_results.to_excel(output_path, index=False)
    if verbose:
        print(f"\nRisultati salvati in: {output_path}")
        print("\n" + "=" * 90)
        print(df_results.to_string(index=False))
        print("=" * 90)

    return df_results


if __name__ == "__main__":
    results = run_tests()

