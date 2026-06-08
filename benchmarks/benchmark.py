"""
Benchmark script to test multiple problem instances in parallel
"""

from src.solver_framework import genetic_solver, merge_solver, ils_solver
from src.utils import check_feasibility
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from time import time
import json
import logging
from tqdm import tqdm
from s339239 import Problem

def solve_single_instance(params):
    """
    Solve a single problem instance with all available solvers.
    
    Args:
        params: dict with keys 'n', 'alpha', 'beta', 'seed', 'density'
    
    Returns:
        dict with results for each solver
    """
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    n, alpha, beta, seed = params['n'], params['alpha'], params['beta'], params['seed']
    density = params.get('density', 0.5)
    
    # Create problem instance
    problem = Problem(num_cities=n, alpha=alpha, beta=beta, seed=seed, density=density)
    
    # Dictionary of solvers
    if problem.graph.number_of_nodes() <= 100:
        solvers = {
            'genetic': genetic_solver,
            'merge': merge_solver,
            'ILS': ils_solver,
        }
    else:
        solvers = {
            'merge': merge_solver,
            'genetic': genetic_solver,
        }

    
    # Test each solver
    results = {}
    for solver_name, solver_func in solvers.items():
        try:
            start_time = time()
            path, cost = solver_func(problem)
            elapsed_time = time() - start_time
            
            feasible = check_feasibility(problem, path)
            
            results[solver_name] = {
                'cost': cost,
                'time': elapsed_time,
                'path_length': len(path),
                'feasible': feasible,
                # Optionally store the path (can be large)
                # 'path': path,
            }
        except Exception as e:
            results[solver_name] = {
                'error': str(e),
                'cost': float('inf'),
                'time': 0,
                'feasible': False
            }
    
    # Get baseline
    baseline = problem.baseline()
    
    # Calculate improvement for each solver
    for solver_name, solver_result in results.items():
        if solver_result.get('feasible', False) and baseline > 0:
            improvement = ((baseline - solver_result['cost']) / baseline) * 100
            solver_result['improvement'] = improvement
        else:
            solver_result['improvement'] = None
    
    # Find best solver
    valid_solvers = {k: v for k, v in results.items() if v.get('feasible', False)}
    if valid_solvers:
        best_solver = min(valid_solvers.items(), key=lambda x: x[1]['cost'])
        best_name = best_solver[0]
    else:
        best_name = None
    
    return {
        'params': params,
        'solvers': results,
        'best_solver': best_name,
        'baseline': baseline
    }


def benchmark_parallel(instances, max_workers=4):
    """
    Run benchmark on multiple instances in parallel.
    
    Args:
        instances: list of dicts with keys 'n', 'alpha', 'beta', 'seed'
        max_workers: number of parallel workers
    
    Returns:
        dict mapping instance params to results
    """
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all instances
        futures = {executor.submit(solve_single_instance, params): params 
                   for params in instances}
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Benchmarking"):
            params = futures[future]
            try:
                result = future.result()
                # Use string key for JSON serialization
                key = f"n={params['n']}_alpha={params['alpha']}_beta={params['beta']}_density={params.get('density', 0.5)}_seed={params['seed']}"
                results[key] = result
                
                logging.info(f"✓ Completed: {key}")
            except Exception as e:
                logging.error(f"✗ Failed: {params} - {e}")
    
    return results


def save_results(results, filename='benchmark_results.json'):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")



def generate_test_instances():
    """Generate a list of test instances"""
    from itertools import product
    from random import randint
    instances = []
    
    # Example: test different combinations
    n_cities = [10, 100, 1000]
    alpha_values = [0.0, 1.0, 2.0]
    beta_values = [0.5, 1.0, 2.0]
    density_values = [0.2, 0.5, 1.0]

    # n_cities = [1000]
    # alpha_values = [1.0]
    # beta_values = [0.5, 1.0, 2.0]
    # density_values = [0.2, 0.5, 1.0]

    
    for n, alpha, beta, density in product(n_cities, alpha_values, beta_values, density_values):
        instances.append({
            'n': n,
            'alpha': alpha,
            'beta': beta,
            'density': density,
            'seed': randint(0, 10000)
        })
    return instances    


def benchmark():
    """
    Do benchmarking of multiple problem instances and save results to a JSON file.
    """
        # Generate test instances
    print("Generating test instances...")
    instances = generate_test_instances()
    print(f"Total instances to test: {len(instances)}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    start_time = time()
    results = benchmark_parallel(instances, max_workers=3)
    total_time = time() - start_time
    
    print(f"\nBenchmark completed in {total_time:.2f}s")
    print(f"Tested {len(results)} instances")
    
    # Save results
    save_results(results, filename='benchmark_results.json')
    
    # Print summary
    print("\n=== Summary ===")
    for key, result in results.items():
        if result.get('best_solver'):
            best = result['solvers'][result['best_solver']]
            improvement = best.get('improvement', 0)
            if improvement is not None:
                print(f"{key}: {result['best_solver']} won with cost {best['cost']:.2f} ({improvement:+.2f}% vs baseline) in {best['time']:.2f}s")
            else:
                print(f"{key}: {result['best_solver']} won with cost {best['cost']:.2f} in {best['time']:.2f}s")

def print_results(path: str):
    """
    Print the results from a benchmark JSON file in a readable format.
    
    :param path: Path to the JSON results file
    :type path: str
    """
    from collections import defaultdict
    
    with open(path, 'r') as f:
        results = json.load(f)
    
    # Count wins for each solver
    wins = defaultdict(int)
    total_instances = 0
    file = open("results_benchmark.txt", "w")
    for key, result in results.items():
        params = result['params']
        best_solver = result['best_solver']
        baseline = result['baseline']
        
        print(f"Instance: n={params['n']}, alpha={params['alpha']}, beta={params['beta']}, seed={params['seed']}")
        print(f"  Baseline cost: {baseline:.2f}")
        
        for solver_name, solver_result in result['solvers'].items():
            cost = solver_result.get('cost', float('inf'))
            time_taken = solver_result.get('time', 0)
            feasible = solver_result.get('feasible', False)
            improvement = solver_result.get('improvement')
            status = "✓" if feasible else "✗"
            
            if improvement is not None:
                print(f"    {solver_name}: cost={cost:.2f}, time={time_taken:.2f}s, improvement={improvement:+.2f}%, feasible={status}")
            else:
                print(f"    {solver_name}: cost={cost:.2f}, time={time_taken:.2f}s, feasible={status}")
        
        if best_solver:
            best_cost = result['solvers'][best_solver]['cost']
            wins[best_solver] += 1
            total_instances += 1
            print(f"  Best solver: {best_solver} with cost {best_cost:.2f}\n")
            file.write(f"Instance: n={params['n']}, alpha={params['alpha']}, beta={params['beta']}, density={params['density']}, seed={params['seed']}, best_solver={best_solver}, best_cost={best_cost:.2f}\n")
        else:
            total_instances += 1
            print("  No feasible solution found by any solver.\n")
    
    # Print win statistics
    print("\n=== Win Statistics ===")
    if wins:
        for solver_name, win_count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
            win_percentage = (win_count / total_instances) * 100 if total_instances > 0 else 0
            print(f"{solver_name}: {win_count}/{total_instances} wins ({win_percentage:.1f}%)")
            file.write(f"{solver_name}: {win_count}/{total_instances} wins ({win_percentage:.1f}%)\n")
    else:
        print("No wins recorded (no feasible solutions found)")

    file.close()

def csv_report(path: str):
    """
    Convert benchmark results from JSON to CSV format.
    """

    with open(path, 'r') as f:
        results = json.load(f)

    with open('benchmark_results.csv', 'w') as f:
        f.write("n,alpha,beta,density,seed,best_solver,best_cost,time,best_improvement,feasible\n")

        for key, result in results.items():
            params = result['params']
            best_solver = result['best_solver']
            if best_solver:
                best_result = result['solvers'][best_solver]
                best_cost = best_result['cost']
                time_taken = best_result['time']
                improvement = best_result.get('improvement', 0)
                feasible = best_result.get('feasible', False)
                
                f.write(f"{params['n']},{params['alpha']},{params['beta']},{params.get('density', 0.5)},{params['seed']},{best_solver},{best_cost},{time_taken},{improvement},{feasible}\n")
        
def latex_results():
    #n,alpha,beta,density,seed,best_solver,best_cost,time,best_improvement,feasible
    import pandas as pd
    df = pd.read_csv('benchmark_results.csv')

    df.drop(columns=['seed', 'feasible'], inplace=True)
    df.rename(columns={
        'n': 'Num Cities',
        'alpha': 'Alpha',
        'beta': 'Beta',
        'density': 'Density',
        'best_solver': 'Best Solver',
        'best_cost': 'Best Cost',
        'time': 'Time (s)',
        'best_improvement': 'Improvement (%)'
    }, inplace=True)

    

    with open('benchmark_results.tex', 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))

import sys

if __name__ == '__main__':
    sys.path.append('..')


    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    benchmark()      # Uncomment to run benchmark
    # print_results('benchmark_results.json')
    # csv_report('benchmark_results.json')  # Uncomment to convert results to CSV
    # latex_results()  # Uncomment to convert CSV results to LaTeX
