import networkx as nx
import numpy as np
import random
import time


def _core_heuristic_optimization(problem, max_iterations=60, time_lim=25.0, k_neighbors=12, random_seed=42):
    random.seed(random_seed)
    G = problem.graph
    t_start = time.time()

    alpha_val, beta_val = problem._alpha, problem._beta

    valid_nodes = [node for node in G.nodes if node != 0]
    num_nodes = len(valid_nodes)

    node_to_id = {node: idx for idx, node in enumerate(valid_nodes)}
    id_to_node = {idx: node for idx, node in enumerate(valid_nodes)}
    depot_index = num_nodes

    distance_matrix = np.full((num_nodes + 1, num_nodes + 1), np.inf, dtype=np.float32)
    temp_mapping = node_to_id.copy()
    temp_mapping[0] = depot_index

    for source, path_lengths in nx.all_pairs_dijkstra_path_length(G, weight="dist"):
        if source not in temp_mapping:
            continue
        src_idx = temp_mapping[source]
        for target, dist_val in path_lengths.items():
            if target in temp_mapping:
                distance_matrix[src_idx, temp_mapping[target]] = dist_val

    gold_attr = nx.get_node_attributes(G, "gold")
    gold_array = np.array([gold_attr[id_to_node[i]] for i in range(num_nodes)])

    def evaluate_trip_cost(sequence):
        total_weight_cost = 0.0
        accumulated_gold = 0.0
        curr_pos = depot_index

        for step in sequence:
            distance = distance_matrix[curr_pos, step]
            total_weight_cost += distance + (distance * alpha_val * accumulated_gold) ** beta_val
            accumulated_gold += gold_array[step]
            curr_pos = step

        dist_home = distance_matrix[curr_pos, depot_index]
        total_weight_cost += dist_home + (dist_home * alpha_val * accumulated_gold) ** beta_val
        return total_weight_cost

    active_trips = {i: [i] for i in range(num_nodes)}
    trip_costs = {i: evaluate_trip_cost([i]) for i in range(num_nodes)}
    node_ownership = {i: i for i in range(num_nodes)}

    candidate_peers = {}
    for i in range(num_nodes):
        sorted_closest = np.argsort(distance_matrix[i, :num_nodes])[1:k_neighbors]
        candidate_peers[i] = list(sorted_closest)
        random.shuffle(candidate_peers[i])

    has_improved = True
    loop_counter = 0

    while has_improved and loop_counter < max_iterations and (time.time() - t_start) < time_lim:
        has_improved = False
        loop_counter += 1

        max_savings = 1e-5
        optimal_move = None

        for trip_id_a in list(active_trips.keys()):
            if trip_id_a not in active_trips:
                continue

            path_a = active_trips[trip_id_a]
            tail_node = path_a[-1]

            for peer in candidate_peers[tail_node]:
                trip_id_b = node_ownership[peer]

                if trip_id_a == trip_id_b or trip_id_b not in active_trips:
                    continue

                path_b = active_trips[trip_id_b]
                if path_b[0] != peer:
                    continue

                real_a = id_to_node[tail_node]
                real_b = id_to_node[peer]

                if not G.has_edge(real_a, real_b):
                    continue

                len_limit = 2 + int(beta_val // 1.5)
                if beta_val < 2:
                    len_limit += 1

                if len(path_a) + len(path_b) > len_limit:
                    continue

                combined_seq = path_a + path_b
                is_valid_sequence = all(
                    G.has_edge(id_to_node[u], id_to_node[v])
                    for u, v in zip(combined_seq, combined_seq[1:])
                )

                if not is_valid_sequence:
                    continue

                updated_cost = evaluate_trip_cost(combined_seq)
                previous_cost = trip_costs[trip_id_a] + trip_costs[trip_id_b]
                savings = previous_cost - updated_cost

                noise_limit = 1e-5 * previous_cost * (1 + 0.2 * random.random())

                if savings > max_savings and savings > noise_limit:
                    max_savings = savings
                    optimal_move = (trip_id_a, trip_id_b, combined_seq, updated_cost)

        if optimal_move:
            t_a, t_b, new_seq, updated_c = optimal_move
            active_trips[t_a] = new_seq
            trip_costs[t_a] = updated_c

            for n in active_trips[t_b]:
                node_ownership[n] = t_a

            del active_trips[t_b]
            del trip_costs[t_b]
            has_improved = True

    global_path = []
    sorted_trips = sorted(active_trips.keys(), key=lambda r: trip_costs[r], reverse=True)

    for t_id in sorted_trips:
        seq = active_trips[t_id]
        if not seq:
            continue

        curr_pos = 0
        sub_path = [(0, 0)]

        for idx in seq:
            real_node = id_to_node[idx]

            if sub_path[-1][0] == real_node:
                continue

            if not G.has_edge(curr_pos, real_node):
                if sub_path[-1][0] != 0:
                    sub_path.append((0, 0))
                curr_pos = 0
                continue

            sub_path.append((real_node, gold_attr[real_node]))
            curr_pos = real_node

        if sub_path[-1][0] != 0:
            sub_path.append((0, 0))

        if global_path and global_path[-1][0] == 0 and sub_path[0][0] == 0:
            sub_path = sub_path[1:]

        global_path.extend(sub_path)

    sanitized_path = [(0, 0)]
    for node_id, gold_val in global_path[1:]:
        prev_node = sanitized_path[-1][0]
        if node_id != prev_node and G.has_edge(prev_node, node_id):
            sanitized_path.append((node_id, gold_val))
        elif node_id == 0:
            sanitized_path.append((0, 0))

    return sanitized_path


def merge_solver(problem, max_iter=60, time_limit=25.0, neighbor_count=12, seed=42) -> tuple[
    list[tuple[int, float]], float]:
    final_path = _core_heuristic_optimization(
        problem,
        max_iterations=max_iter,
        time_lim=time_limit,
        k_neighbors=neighbor_count,
        random_seed=seed
    )

    cost = 0.0
    current_gold = 0.0
    alpha_val, beta_val = problem._alpha, problem._beta
    graph = problem.graph

    # -----------------------------------------------------
    # CALCOLO LOG INTERNO ALLINEATO ALLA VALUTAZIONE
    # Usa shortest_path per gestire i "salti", eliminando il KeyError
    # ma senza toccare il percorso reale!
    # -----------------------------------------------------
    for i in range(len(final_path) - 1):
        u = final_path[i][0]
        v = final_path[i + 1][0]
        current_gold += final_path[i][1]

        try:
            sp = nx.shortest_path(graph, u, v, weight="dist")
            d = nx.path_weight(graph, sp, weight="dist")
        except nx.NetworkXNoPath:
            d = float("inf")

        cost += d + (d * alpha_val * current_gold) ** beta_val

        if v == 0:
            current_gold = 0.0

    return final_path, cost