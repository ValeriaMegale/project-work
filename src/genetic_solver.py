import random
import copy
import time
import networkx as nx
import numpy as np
from .individual import Individual
from Problem import Problem
import scipy.sparse.csgraph as csgraph

class GeneticSolver:
    def __init__(self, problem: Problem, pop_size=100, generations=100, mutation_rate=0.3, elite_size=2):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        nodes = list(problem.graph.nodes)
        # print(f"🧬 GA: Precomputing distance matrix for {len(nodes)} nodes using scipy...")
        t0 = time.time()
        adj_mat = nx.to_scipy_sparse_array(problem.graph, nodelist=nodes, weight='dist')
        dist_matrix = csgraph.shortest_path(adj_mat, method='D', directed=True)
        # print(f"🧬 GA: Distance matrix precomputed in {time.time()-t0:.2f}s")
        
        self.dist_cache = {
            u: {v: d for v, d in zip(nodes, row) if not np.isinf(d)}
            for u, row in zip(nodes, dist_matrix)
        }
        
        # Shared cache for shortest paths (used in rebuild_phenotype)
        self.path_cache = {}
        
        self.cities = list(problem.graph.nodes)
        if 0 in self.cities:
            self.cities.remove(0)

    def create_individual(self, genome=None):
        if genome is None:
            genome = random.sample(self.cities, len(self.cities))
        ind = Individual(genome, self.problem, self.dist_cache, self.path_cache)
        ind.evaluate()
        return ind

    def crossover(self, p1, p2):
        size = len(p1.genome)
        start, end = sorted(random.sample(range(size), 2))
        child_genome = [None] * size
        child_genome[start:end+1] = p1.genome[start:end+1]

        # O(n) set lookup instead of O(n) list scan → crossover O(n) instead of O(n²)
        inherited = set(p1.genome[start:end+1])

        current_idx = (end + 1) % size
        p2_idx = (end + 1) % size

        remaining = size - (end - start + 1)
        while remaining > 0:
            gene = p2.genome[p2_idx]
            if gene not in inherited:
                child_genome[current_idx] = gene
                inherited.add(gene)
                current_idx = (current_idx + 1) % size
                remaining -= 1
            p2_idx = (p2_idx + 1) % size

        return Individual(child_genome, self.problem, self.dist_cache, self.path_cache)

    def evolve(self):
        start_time = time.time()
        time_limit = 570  # 9 minutes 30 seconds
        
        # print(f"🧬 GA: Starting evolution with pop={self.pop_size}, gen={self.generations}, n_cities={len(self.cities)}")
        population = []
        
        # --- HEURISTIC INJECTION (When Density=1) ---
        # 1. Create a "Dijkstra-like" individual: visits cities from closest to farthest
        smart_genome = sorted(self.cities, key=lambda c: self.dist_cache[0][c])
        population.append(self.create_individual(smart_genome))
        
        # Fill the rest of the population with random individuals
        for _ in range(self.pop_size - 1):
            population.append(self.create_individual())
            
        best_overall = min(population, key=lambda x: x.fitness)
        
        stagnation_counter = 0
        current_best_fitness = best_overall.fitness
        base_mutation_rate = self.mutation_rate
        
        for g in range(self.generations):
            gen_start = time.time()
            # Hard early stopping at 9:30 minutes
            if time.time() - start_time > time_limit:
                # print(f"🧬 GA: Time limit reached at gen {g}")
                break
            population.sort(key=lambda x: x.fitness)
            
            # Elitism: keep track of the best individual found so far
            if population[0].fitness < best_overall.fitness:
                best_overall = copy.deepcopy(population[0])
                stagnation_counter = 0
                self.mutation_rate = base_mutation_rate # Reset mutation rate
                current_best_fitness = population[0].fitness
            else:
                stagnation_counter += 1
            
            # --- ADAPTIVE MUTATION ---
            # If we have stagnated for more than 5 generations, increase mutation rate
            if stagnation_counter > 15:
                self.mutation_rate = min(0.8, base_mutation_rate * 3) # Hyper-mutation
            elif stagnation_counter > 5:
                self.mutation_rate = min(0.5, base_mutation_rate * 1.5)

            new_population = population[:self.elite_size]
            
            while len(new_population) < self.pop_size:
                candidates = random.sample(population, 3)
                parent1 = min(candidates, key=lambda x: x.fitness)
                candidates = random.sample(population, 3)
                parent2 = min(candidates, key=lambda x: x.fitness)
                
                child = self.crossover(parent1, parent2)
                
                child.mutate(self.mutation_rate)
                
                # --- MEMETIC ALGORITHM (Local Search) ---
                # Fast local optimization to refine the child
                child.local_optimize()

                # If fitness is still infinite, evaluate it
                # (this can happen if local_optimize doesn't update fitness)
                if child.fitness == float('inf'):
                    child.evaluate()
                    
                new_population.append(child)
            
            population = new_population
            
            if g % 5 == 0 or g < 3:
                # print(f"🧬 GA: Gen {g}/{self.generations} | best={best_overall.fitness:.2f} | gen_time={time.time()-gen_start:.2f}s | mut_rate={self.mutation_rate:.2f}")
                pass

        # print(f"🧬 GA: Evolution completed in {time.time()-start_time:.2f}s, best_fitness={best_overall.fitness:.2f}")
        return best_overall