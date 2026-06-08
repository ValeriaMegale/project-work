import networkx as nx
import math
from typing import List, Tuple, Dict
from time import time
import logging
from Problem import Problem

def get_return_path_optimized(G: nx.Graph, start_node: int, current_weight: float, alpha: float, beta: float) -> Tuple[float, List[int]]:
    """
    Finds the optimal weighted path back to depot (node 0) using A* search.
    
    Args:
        G: The graph
        start_node: Current location
        current_weight: Gold currently carried
        alpha, beta: Problem parameters for cost calculation
    
    Returns:
        (total_cost, path_nodes)
    """
    target_node = 0
    target_pos = G.nodes[target_node]['pos']

    def heuristic(u, v):
        # Euclidean distance lower bound for weighted travel cost
        u_pos = G.nodes[u]['pos']
        dist_geom = math.dist(u_pos, target_pos)
        return dist_geom + (dist_geom * alpha * current_weight) ** beta

    def weight_fn(u, v, d):
        # Actual edge cost with weight penalty
        dist_edge = d['dist']
        return dist_edge + (dist_edge * alpha * current_weight) ** beta

    try:
        path = nx.astar_path(G, source=start_node, target=target_node, 
                            heuristic=heuristic, weight=weight_fn)
        
        # Calculate total path cost
        cost = sum(weight_fn(path[i], path[i+1], G[path[i]][path[i+1]]) 
                   for i in range(len(path) - 1))
        return cost, path
        
    except nx.NetworkXNoPath:
        return float('inf'), []


def single_paths(problem: Problem, src=0) -> Dict[int, dict]:
    """
    Computes optimal single-city round trips (depot -> city -> depot).
    Each trip picks up gold at one city only.
    
    Returns:
        Dictionary mapping city_id to trip info (path, cost, gold)
    """
    G = problem.graph
    alpha = problem._alpha
    beta = problem._beta
    gold_at = nx.get_node_attributes(G, "gold")
    
    # Compute shortest outbound paths (unweighted) from depot
    dist_out, paths_out = nx.single_source_dijkstra(G, source=src, weight='dist')
    
    single_solutions = {}
    cnt = 0
    total_cities = len(gold_at)

    for dst, gold in gold_at.items():
        if cnt % 100 == 0:
            logging.debug(f"[single_paths] Processing city {cnt}/{total_cities}", end='\r')
        cnt += 1
        
        if dst == src or gold <= 0:
            continue
            
        path_go = paths_out[dst]
        
        # Find optimal return path with weight using A*
        cost_ret, path_ret = get_return_path_optimized(G, dst, gold, alpha, beta)
        
        # Build full round trip: go empty, pick up gold, return weighted
        full_path = [(n, 0.0) for n in path_go[:-1]]
        full_path.append((dst, gold))
        full_path.extend([(n, 0.0) for n in path_ret[1:]])
        
        total_cost = dist_out[dst] + cost_ret

        single_solutions[dst] = {
            'gold_picked': {dst},
            'current_weight': gold,
            'path': full_path,
            'cost': total_cost,
            'original_single_cost': total_cost
        }
        
    logging.debug(f"[single_paths] Processing complete.")
    return single_solutions


def fast_cost_calc(distances_cache: List[float], start_idx: int, current_weight: float, alpha: float, beta: float) -> float:
    """
    Efficiently calculates path cost from start_idx to end using pre-cached distances.
    """
    cost = 0.0
    
    for i in range(start_idx, len(distances_cache)):
        dist = distances_cache[i]
        if current_weight > 0:
            cost += dist + pow(dist * alpha * current_weight, beta)
        else:
            cost += dist
    return cost


def merge_strategy_optimized(problem: Problem) -> List[List[Tuple[int, float]]]:
    """
    Merge strategy: combines multiple cities into single trips when profitable.
    
    Algorithm:
    1. Generate all single-city round trips
    2. Sort by path length (longest first)
    3. For each main trip, try merging other cities that lie on the return path
    4. Merge if marginal cost < standalone cost of candidate city
    
    Returns:
        List of trips, where each trip is [(city, gold_picked), ...]
    """
    G = problem.graph
    alpha = problem._alpha
    beta = problem._beta
    
    start_time = time()
    paths_info = single_paths(problem)
    end_time = time()
    logging.debug(f"[merge_strategy_optimized] single_paths computed in {end_time - start_time:.2f} seconds.")
    
    excluded_cities = set()  # Cities already merged into other trips
    
    # Process longer trips first (more opportunities for merging)
    sorted_destinations = sorted(paths_info.keys(), 
                               key=lambda k: len(paths_info[k]['path']), 
                               reverse=True)
    
    final_solution_paths = []

    start_time = time()
    for main_dst in sorted_destinations:
        if main_dst in excluded_cities:
            continue

        excluded_cities.add(main_dst)

        current_data = paths_info[main_dst]
        full_path = current_data['path']
        
        # Pre-cache edge distances for fast cost recalculation
        path_distances = [G[full_path[k][0]][full_path[k+1][0]]['dist'] 
                         for k in range(len(full_path) - 1)]
            
        current_vehicle_weight = current_data['current_weight']
        last_pick_idx = max(i for i, (_, g) in enumerate(full_path) if g > 0)
        
        # Find cities on return path that could be merged
        candidate_indices = []
        for i in range(last_pick_idx + 1, len(full_path) - 1):
            city = full_path[i][0]
            if city in paths_info and city not in excluded_cities and city != main_dst:
                candidate_indices.append(i)
        
        # Try merging each candidate
        for i in candidate_indices:
            candidate_city = full_path[i][0]
            if candidate_city in excluded_cities:
                continue

            cost_candidate_solo = paths_info[candidate_city]['original_single_cost']
            gold_candidate = paths_info[candidate_city]['current_weight']
            
            new_weight = current_vehicle_weight + gold_candidate
            
            # Compare cost of returning with extra gold vs current weight
            cost_return_heavy = fast_cost_calc(path_distances, i, new_weight, alpha, beta)
            cost_return_light = fast_cost_calc(path_distances, i, current_vehicle_weight, alpha, beta)
            
            marginal_cost = cost_return_heavy - cost_return_light
            
            # Merge if cheaper than doing candidate city separately
            if marginal_cost < cost_candidate_solo:
                full_path[i] = (candidate_city, gold_candidate)
                current_vehicle_weight += gold_candidate
                excluded_cities.add(candidate_city)

        final_solution_paths.append(full_path)
        
    end_time = time()
    logging.debug(f"[merge_strategy_optimized] Merging completed in {end_time - start_time:.2f} seconds.")
             
    return final_solution_paths

