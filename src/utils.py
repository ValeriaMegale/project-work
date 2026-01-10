from typing import List, Tuple
import networkx as nx
from src.beta_optimizer import path_optimizer
from Problem import Problem

def check_feasibility(
    problem: Problem,
    solution: List[Tuple[int, float]],
) -> bool:
    """
    Checks if a solution is feasible:
    1. Each step must be between adjacent cities
    2. All gold from all cities must be collected (at least once)
    
    :param problem: Problem instance
    :param solution: List of (city, gold_picked)
    :return: True if feasible, False otherwise
    """
    graph = problem.graph
    gold_at = nx.get_node_attributes(graph, "gold")
    
    # Track collected gold per city
    gold_collected = {}
    prev_city = 0  # Start from depot
    
    current_weight = 0
    
    for city, gold in solution[1:]:
        # Check adjacency
        if not graph.has_edge(prev_city, city):
            print(f"❌ Feasibility failed: no edge between {prev_city} and {city}")
            return False
        
        # Track collected gold
        if gold > 0:
            gold_collected[city] = gold_collected.get(city, 0.0) + gold
        
        # Update current weight
        current_weight += gold
        if city == 0:
            current_weight = 0
            
        prev_city = city
    
    # Verify all gold was collected
    for city in graph.nodes():
        if city == 0:  # Depot has no gold
            continue
        expected_gold = gold_at.get(city, 0.0)
        collected_gold = gold_collected.get(city, 0.0)
        
        if abs(expected_gold - collected_gold) > 1e-4:  # Float tolerance
            print(f"❌ Feasibility failed: city {city} has {expected_gold:.2f} gold, collected {collected_gold:.2f}")
            return False
    
    return True

def split_path(path: list[tuple[int, float]]) -> list[list[tuple[int, float]]]:
    """
    Splits a path into sub-paths at each return to depot (node 0).
    
    :param path: Sequence of (city, gold to pick up at city)
                        Example: [(0, 0), (20, 1000), (0, 0), (30, 500), (0, 0)]
        :type path: list[tuple[int, float]]
    :return: List of sub-paths
    :rtype: list[list[tuple[int, float]]]
    """
    sub_paths = []
    current_sub_path = []
    
    for node, gold in path:
        current_sub_path.append((node, gold))
        if node == 0 and len(current_sub_path) > 1:
            sub_paths.append(current_sub_path)
            current_sub_path = [(0, 0.0)]  # Start new sub-path from depot
    
    if len(current_sub_path) > 1:
        sub_paths.append(current_sub_path)
    
    return sub_paths

def join_paths(paths: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    """
    Joins multiple sub-paths into a single path.
    
    :param paths: List of sub-paths
    :type paths: list[list[tuple[int, float]]]
    :return: Joined path
    :rtype: list[tuple[int, float]]
    """
    joined_path = []
    
    for sub_path in paths:
        if joined_path:
            joined_path.pop()  # Remove last depot to avoid duplication
        joined_path.extend(sub_path)
    
    return joined_path

def optimize_full_path(path: list[tuple[int, float]], problem) -> list[tuple[int, float]]:
    """
    Optimizes a single path using beta optimization.
    
    :param path: Sequence of (city, gold to pick up at city)
                        Example: [(0, 0), (20, 1000), (0, 0)]
        :type path: list[tuple[int, float]]
    :param problem: Problem instance
    :type problem: Problem
    :return: Optimized path
    """
    splitted_paths = split_path(path)
    optimized_paths = [path_optimizer(sub_path, problem) for sub_path in splitted_paths]
    return join_paths(optimized_paths)