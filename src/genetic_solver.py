import random
import copy
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
        adj_mat = nx.to_scipy_sparse_array(problem.graph, nodelist=nodes, weight='dist')
        dist_matrix = csgraph.shortest_path(adj_mat, method='D', directed=True)
        
        self.dist_cache = {
            u: {v: d for v, d in zip(nodes, row) if not np.isinf(d)}
            for u, row in zip(nodes, dist_matrix)
        }
        
        self.cities = list(problem.graph.nodes)
        if 0 in self.cities:
            self.cities.remove(0)

    def create_individual(self, genome=None):
        if genome is None:
            genome = random.sample(self.cities, len(self.cities))
        # Rimosso path_cache dai parametri
        ind = Individual(genome, self.problem, self.dist_cache)
        ind.evaluate()
        return ind

    def crossover(self, p1, p2):
        size = len(p1.genome)
        start, end = sorted(random.sample(range(size), 2))
        child_genome = [None] * size
        child_genome[start:end+1] = p1.genome[start:end+1]
        
        current_idx = (end + 1) % size
        p2_idx = (end + 1) % size
        
        while None in child_genome:
            gene = p2.genome[p2_idx]
            if gene not in child_genome:
                child_genome[current_idx] = gene
                current_idx = (current_idx + 1) % size
            p2_idx = (p2_idx + 1) % size
            
        return Individual(child_genome, self.problem, self.dist_cache)

    def evolve(self):
        population = []
        
        # --- HEURISTIC INJECTION (Aiuto per Density=1) ---
        # 1. Creiamo un individuo "Dijkstra-like": visita le città dalla più vicina alla più lontana
        smart_genome = sorted(self.cities, key=lambda c: self.dist_cache[0][c])
        population.append(self.create_individual(smart_genome))
        
        # Riempiamo il resto a caso
        for _ in range(self.pop_size - 1):
            population.append(self.create_individual())
            
        best_overall = min(population, key=lambda x: x.fitness)
        
        # Variabili per Mutazione Adattiva
        stagnation_counter = 0
        current_best_fitness = best_overall.fitness
        base_mutation_rate = self.mutation_rate
        
        for g in range(self.generations):
            population.sort(key=lambda x: x.fitness)
            
            # Elitismo: salviamo il migliore assoluto
            if population[0].fitness < best_overall.fitness:
                best_overall = copy.deepcopy(population[0])
                stagnation_counter = 0
                self.mutation_rate = base_mutation_rate # Reset mutation rate
                current_best_fitness = population[0].fitness
            else:
                stagnation_counter += 1
            
            # --- ADAPTIVE MUTATION ---
            # Se siamo bloccati da troppo tempo, aumentiamo la mutazione per uscire dal minimo locale
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
                # Applichiamo una veloce ottimizzazione locale
                child.local_optimize()
                
                # Se local_optimize non ha chiamato evaluate (nessun miglioramento), 
                # assicuriamoci che la fitness sia valida, altrimenti evaluate() è già stato chiamato
                if child.fitness == float('inf'):
                    child.evaluate()
                    
                new_population.append(child)
            
            population = new_population

        return best_overall