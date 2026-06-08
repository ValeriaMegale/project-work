import random
import networkx as nx
from Problem import Problem

class Individual:
    def __init__(self, genome, problem: Problem, dist_cache, path_cache):
        self.genome = genome            
        self.problem = problem
        self.dist_cache = dist_cache
        self.path_cache = path_cache  # Shared cache for shortest paths
        self.fitness = float('inf')
        self.phenotype = []             

    def evaluate(self):
        current_node = 0
        current_weight = 0
        total_cost = 0.0

        # Local references avoid repeated attribute lookups in the hot loop
        alpha = self.problem.alpha
        beta = self.problem.beta
        dist = self.dist_cache
        gold_attr = self.problem.graph.nodes
        genome = self.genome
        dist_from_0 = dist[0]

        for next_node in genome:
            gold = gold_attr[next_node]['gold']

            try:
                d_from_cur = dist[current_node]
                d_dir = d_from_cur[next_node]
                d_home = d_from_cur[0]
                d_out = dist_from_0[next_node]
            except KeyError:
                self.fitness = float('inf')
                return float('inf')

            # Direct cost
            cost_direct = d_dir + (alpha * d_dir * current_weight) ** beta

            # Split cost (return to depot, then go out empty)
            cost_split = d_home + (alpha * d_home * current_weight) ** beta + d_out

            if cost_split < cost_direct:
                total_cost += cost_split
                current_weight = 0
            else:
                total_cost += cost_direct

            current_weight += gold
            current_node = next_node

        d_end = dist[current_node][0]
        total_cost += d_end + (alpha * d_end * current_weight) ** beta

        self.fitness = total_cost
        return total_cost

    def rebuild_phenotype(self):
        """
        Ricostruisce il percorso come lista di tuple [(nodo, oro_preso), ...]
        Ora usa A* con euristica euclidea + cache per efficienza.
        """
        import time
        import math
        t_start = time.time()
        path_calls = 0
        cache_hits = 0
        
        path = []
        current_node = 0
        current_weight = 0
        alpha = self.problem.alpha
        beta = self.problem.beta
        graph = self.problem.graph
        
        # Check if graph has positions for A* heuristic
        has_pos = 'pos' in graph.nodes[0]
        
        def get_path(u, v):
            """Get shortest path with cache and A* if coordinates available."""
            nonlocal path_calls, cache_hits
            key = (u, v)
            if key in self.path_cache:
                cache_hits += 1
                return self.path_cache[key]
            
            path_calls += 1
            if has_pos:
                # A* with euclidean heuristic
                def heuristic(a, b):
                    pos_a = graph.nodes[a]['pos']
                    pos_b = graph.nodes[b]['pos']
                    return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
                
                result = nx.astar_path(graph, u, v, heuristic=heuristic, weight='dist')
            else:
                # Fallback to Dijkstra
                result = nx.shortest_path(graph, u, v, weight='dist')
            
            self.path_cache[key] = result
            return result

        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            d_dir = self.dist_cache[current_node][next_node]
            d_home = self.dist_cache[current_node][0]
            d_out = self.dist_cache[0][next_node]

            cost_direct = d_dir + (alpha * d_dir * current_weight) ** beta
            cost_split = (d_home + (alpha * d_home * current_weight) ** beta) + \
                         (d_out + (alpha * d_out * 0) ** beta)

            if cost_split < cost_direct:
                # 1. Torna a casa (scarica)
                p_home = get_path(current_node, 0)
                for step in p_home[1:-1]:
                    path.append((step, 0))
                path.append((0, 0))
                
                # 2. Ripartire verso next_node
                p_next = get_path(0, next_node)
                for step in p_next[1:-1]:
                    path.append((step, 0))
                
                current_weight = 0
            else:
                # Vai diretto (passi intermedi)
                p_dir = get_path(current_node, next_node)
                for step in p_dir[1:-1]:
                    path.append((step, 0))
            
            # Arrivo alla destinazione e PRENDO L'ORO
            path.append((next_node, gold))
            
            current_weight += gold
            current_node = next_node
        
        # Ritorno finale a 0
        p_end = get_path(current_node, 0)
        for step in p_end[1:]:
            path.append((step, 0))
            
        self.phenotype = path
        elapsed = time.time() - t_start
        if elapsed > 0.5 or len(self.genome) > 200:
            # print(f"🧬 rebuild_phenotype: {len(self.genome)} cities, {path_calls} path computations, {cache_hits} cache hits, {elapsed:.2f}s")
            pass
        return path

    def mutate(self, mutation_rate=0.1):
        """
        Applica mutazione con probabilità mutation_rate.
        Supporta Inversion (2-opt) e Swap.
        """
        if random.random() < mutation_rate:
            # 50% probabilità di Inversion, 50% di Swap
            if random.random() < 0.5:
                # Inversion Mutation (ottima per TSP geometrici)
                idx1, idx2 = random.sample(range(len(self.genome)), 2)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                self.genome[idx1:idx2+1] = self.genome[idx1:idx2+1][::-1]
            else:
                # Swap Mutation (ottima per riordinare singole città)
                idx1, idx2 = random.sample(range(len(self.genome)), 2)
                self.genome[idx1], self.genome[idx2] = self.genome[idx2], self.genome[idx1]

    def local_optimize(self):
        """
        Algoritmo Memetico: Ricerca Locale (Hill Climbing).
        Prova scambi adiacenti su un campione casuale di posizioni.
        Per n grande, limita drasticamente il lavoro.
        """
        improved = False
        current_fitness = self.fitness
        n = len(self.genome)

        # Scala il budget: per n=50 → 50, per n=1000 → ~20
        max_iter = min(20, n - 1) if n > 100 else min(50, n - 1)

        # Per n grandi, campiona posizioni casuali invece di scansione sequenziale
        if n > 100:
            positions = random.sample(range(n - 1), max_iter)
        else:
            positions = range(max_iter)

        for i in positions:
            self.genome[i], self.genome[i+1] = self.genome[i+1], self.genome[i]

            new_fitness = self.evaluate()

            if new_fitness < current_fitness:
                current_fitness = new_fitness
                improved = True
            else:
                self.genome[i], self.genome[i+1] = self.genome[i+1], self.genome[i]
                self.fitness = current_fitness

        return improved