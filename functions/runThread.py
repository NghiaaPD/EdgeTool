import streamlit as st

from lib.edgeAlgorithm import edge_computing_cost_with_priority, PSO, genetic_algorithm, NSGAII


def run_edge_computing_cost_with_priority(queue, positions, distances, processing_powers, task_sizes, priorities):
    random_cost = edge_computing_cost_with_priority(positions, distances, processing_powers, task_sizes, priorities)
    queue.put(("Edge computing cost with priority", random_cost))

def run_PSO(queue, cost_func, num_particles, num_iterations, dim, bounds):
    result = PSO(cost_func=cost_func, num_particles=num_particles, num_iterations=num_iterations, dim=dim, bounds=bounds)
    best_pso_position, best_pso_cost = result.optimize()
    queue.put(("PSO", best_pso_cost))
def run_genetic_algorithm(queue, num_jobs, num_edge_devices, distances, processing_powers, task_sizes, priorities,
                          pop_size, num_generations, crossover_rate, mutation_rate):
    best_ga_position, best_ga_cost = genetic_algorithm(num_jobs, num_edge_devices, distances, processing_powers,
                                                       task_sizes, priorities, pop_size, num_generations,
                                                       crossover_rate, mutation_rate)

    queue.put(("GA", best_ga_cost))
def run_NSGAII(queue, distances,
               processing_powers,
               task_sizes,
               priorities,
               num_jobs,
               num_edge_devices,
               pop_size,
               num_generations):
    nsga2 = NSGAII(pop_size=pop_size, num_generations=num_generations, distances=distances, processing_powers=processing_powers,
                   task_sizes=task_sizes, priorities=priorities, num_jobs=num_jobs,
                   num_edge_devices=num_edge_devices)

    best_nsga2_population, best_nsga2_costs = nsga2.run()
    queue.put(("NSGAII", min(best_nsga2_costs[:, 0])))

