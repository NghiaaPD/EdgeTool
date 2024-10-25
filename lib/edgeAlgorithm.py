import numpy as np
import random


# ---- PSO Algorithm ----
class PSO:
    def __init__(self, cost_func, num_particles, num_iterations, dim, bounds, w=0.5, c1=1, c2=2):
        self.cost_func = cost_func
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = dim
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.positions = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_particles, dim))
        self.velocities = np.random.uniform(low=-1, high=1, size=(num_particles, dim))
        self.p_best = self.positions.copy()
        self.g_best = np.zeros(dim)
        self.g_best_cost = float('inf')
        self.p_best_cost = np.array([float('inf') for _ in range(num_particles)])

    def optimize(self):
        for t in range(self.num_iterations):
            for i in range(self.num_particles):
                cost = self.cost_func(self.positions[i])
                if cost < self.p_best_cost[i]:
                    self.p_best[i] = self.positions[i]
                    self.p_best_cost[i] = cost
                if cost < self.g_best_cost:
                    self.g_best = self.positions[i]
                    self.g_best_cost = cost

            r1, r2 = np.random.random(size=(2, self.num_particles, self.dim))
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.p_best - self.positions) +
                               self.c2 * r2 * (self.g_best - self.positions))
            self.positions = self.positions + self.velocities
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

        return self.g_best, self.g_best_cost


# ---- GA Algorithm ----
def edge_computing_cost_with_priority(positions, distances, processing_powers, task_sizes, priorities):
    num_jobs = positions.shape[0]
    total_cost = 0
    for i in range(num_jobs):
        edge_idx = positions[i]
        distance_cost = distances[i][edge_idx]
        # print(distance_cost, type(distance_cost))
        computation_cost = task_sizes[i] / processing_powers[edge_idx]
        total_cost += priorities[i] * (distance_cost + computation_cost)
    return total_cost

def initialize_population(pop_size, num_jobs, num_edge_devices):
    return np.random.randint(0, num_edge_devices, size=(pop_size, num_jobs))


def calculate_fitness(population, distances, processing_powers, task_sizes, priorities):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fitness[i] = edge_computing_cost_with_priority(population[i], distances, processing_powers, task_sizes,
                                                       priorities)
    return fitness


def tournament_selection(population, fitness, tournament_size=5):
    selected_indices = np.random.choice(np.arange(population.shape[0]), tournament_size)
    best_idx = selected_indices[np.argmin(fitness[selected_indices])]
    return population[best_idx]


def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2


def mutate(individual, mutation_rate, num_edge_devices):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(0, num_edge_devices)
    return individual


def genetic_algorithm(num_jobs, num_edge_devices, distances, processing_powers, task_sizes, priorities,
                      pop_size=50, num_generations=100, crossover_rate=0.8, mutation_rate=0.02):
    population = initialize_population(pop_size, num_jobs, num_edge_devices)
    fitness = calculate_fitness(population, distances, processing_powers, task_sizes, priorities)

    for generation in range(num_generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            child1 = mutate(child1, mutation_rate, num_edge_devices)
            child2 = mutate(child2, mutation_rate, num_edge_devices)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        population = np.array(new_population)
        fitness = calculate_fitness(population, distances, processing_powers, task_sizes, priorities)
        # print(f"Generation {generation+1}, Best fitness: {np.min(fitness)}")

    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    best_cost = fitness[best_idx]
    return best_individual, best_cost


# ---- NSGA-II Algorithm ----
class NSGAII:
    def __init__(self, pop_size, num_generations, distances, processing_powers, task_sizes, priorities, num_jobs,
                 num_edge_devices):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.distances = distances
        self.processing_powers = processing_powers
        self.task_sizes = task_sizes
        self.priorities = priorities
        self.num_jobs = num_jobs
        self.num_edge_devices = num_edge_devices

        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.randint(0, self.num_edge_devices, size=(self.pop_size, self.num_jobs))

    def evaluate(self, population):
        costs = np.zeros((population.shape[0], 2))
        for i in range(population.shape[0]):
            edge_idx = population[i]
            distance_cost = sum(self.distances[j][edge_idx[j]] for j in range(self.num_jobs))
            computation_cost = sum(
                self.task_sizes[j] / self.processing_powers[edge_idx[j]] for j in range(self.num_jobs))
            costs[i] = [distance_cost, computation_cost]
        return costs

    def non_dominated_sort(self, costs):
        num_solutions = costs.shape[0]
        dominated_count = np.zeros(num_solutions)
        dominance_set = [[] for _ in range(num_solutions)]

        for p in range(num_solutions):
            for q in range(num_solutions):
                if all(costs[p] <= costs[q]) and any(costs[p] < costs[q]):
                    dominance_set[p].append(q)
                elif all(costs[q] <= costs[p]) and any(costs[q] < costs[p]):
                    dominated_count[p] += 1

        fronts = []
        current_front = []

        for i in range(num_solutions):
            if dominated_count[i] == 0:
                current_front.append(i)

        fronts.append(current_front)

        while current_front:
            next_front = []
            for p in current_front:
                for q in dominance_set[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        next_front.append(q)
            current_front = next_front
            if next_front:
                fronts.append(next_front)

        return fronts

    def crowding_distance(self, costs, front):
        distances = np.zeros(len(front))
        if len(front) == 0:
            return distances

        for i in range(costs.shape[1]):
            front_costs = costs[front, i]
            sorted_indices = np.argsort(front_costs)
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
            for j in range(1, len(front) - 1):
                distances[sorted_indices[j]] += (
                            front_costs[sorted_indices[j + 1]] - front_costs[sorted_indices[j - 1]])

        return distances

    def select_parents(self, population, costs):
        fronts = self.non_dominated_sort(costs)
        next_population = []
        for front in fronts:
            if len(next_population) + len(front) > self.pop_size:
                distances = self.crowding_distance(costs, front)
                sorted_front = np.array(front)[np.argsort(distances)[::-1]]
                next_population.extend(sorted_front[:self.pop_size - len(next_population)])
                break
            next_population.extend(front)

        return population[next_population]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, individual, mutation_rate):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = np.random.randint(0, self.num_edge_devices)
        return individual

    def run(self):
        for generation in range(self.num_generations):
            costs = self.evaluate(self.population)
            parents = self.select_parents(self.population, costs)
            offspring = []
            while len(offspring) < self.pop_size:
                parent1 = parents[np.random.randint(0, len(parents))]
                parent2 = parents[np.random.randint(0, len(parents))]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, mutation_rate=0.02)
                child2 = self.mutate(child2, mutation_rate=0.02)
                offspring.append(child1)
                if len(offspring) < self.pop_size:
                    offspring.append(child2)
            self.population = np.array(offspring)

        costs = self.evaluate(self.population)
        return self.population, costs


# # ---- Common Data Setup ----
# num_jobs = 500
# num_edge_devices = 131312
#
# # Giả lập khoảng cách từ mỗi tác vụ đến mỗi edge device. Chọn 1 điểm
# distances = np.random.uniform(low=1, high=100, size=(num_jobs, num_edge_devices))
# print(distances, type(distances))
# # Giả lập khả năng tính toán của mỗi edge device (đơn vị là FLOPS)
# processing_powers = np.random.uniform(low=10, high=100, size=num_edge_devices)
# # Giả lập kích thước của mỗi tác vụ (task sizes)
# task_sizes = np.random.uniform(low=100, high=1000, size=num_jobs)
# # Mức độ ưu tiên của các tác vụ (từ 1 đến 10, với 10 là ưu tiên cao nhất)
# priorities = np.random.uniform(low=1, high=10, size=num_jobs)
#
# # ---- Random Allocation ----
# random_allocation = np.random.randint(0, num_edge_devices, size=num_jobs)
# # print("Random Allocation:", random_allocation)
#
# # Tính toán chi phí cho phân bổ ngẫu nhiên
# random_cost = edge_computing_cost_with_priority(random_allocation, distances, processing_powers, task_sizes, priorities)
# print("\nRandom Allocation Cost:", random_cost)
#
# # ---- PSO Execution ----
# pso = PSO(
#     cost_func=lambda pos: edge_computing_cost_with_priority(np.round(pos).astype(int), distances, processing_powers,
#                                                             task_sizes, priorities),
#     num_particles=30, num_iterations=50, dim=num_jobs, bounds=[0, num_edge_devices - 1])
#
# best_pso_position, best_pso_cost = pso.optimize()
#
# print("\nBest PSO Cost:", best_pso_cost)
#
# # ---- GA Execution ----
# best_ga_position, best_ga_cost = genetic_algorithm(num_jobs=num_jobs, num_edge_devices=num_edge_devices,
#                                                    distances=distances, processing_powers=processing_powers,
#                                                    task_sizes=task_sizes, priorities=priorities,
#                                                    pop_size=50, num_generations=50, crossover_rate=0.8,
#                                                    mutation_rate=0.02)
#
# print("\nBest GA Cost:", best_ga_cost)
#
# # ---- NSGA-II Execution ----
# nsga2 = NSGAII(pop_size=50, num_generations=50, distances=distances, processing_powers=processing_powers,
#                task_sizes=task_sizes, priorities=priorities, num_jobs=num_jobs, num_edge_devices=num_edge_devices)
#
# best_nsga2_population, best_nsga2_costs = nsga2.run()
#
# print("\nNSGA2:", min(best_nsga2_costs[:, 0]))
# # print("NSGA-II Cost for each solution:")
# # for cost in best_nsga2_costs:
# #     print(f"Distance Cost: {cost[0]}, Computation Cost: {cost[1]}")
#
# # ---- Compare Results ----
# if best_pso_cost < best_ga_cost and best_pso_cost < min(best_nsga2_costs[:, 0]):
#     print("\nPSO performed best with the lowest cost.")
# elif best_ga_cost < best_pso_cost and best_ga_cost < min(best_nsga2_costs[:, 0]):
#     print("\nGA performed best with the lowest cost.")
# else:
#     print("\nNSGA-II performed best with the lowest cost.")
