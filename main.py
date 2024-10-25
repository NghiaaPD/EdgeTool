import streamlit as st
import pandas as pd
import threading
from queue import Queue

from functions.loadData import Load
from functions.mapData import Map
from functions.runThread import *

from config.configParameter import *

from lib.edgeAlgorithm import edge_computing_cost_with_priority, PSO, genetic_algorithm, NSGAII

st.set_page_config(page_title="Main", page_icon="📈")

with st.sidebar:
    option = st.selectbox(
        "Select an algorithm",
        ("Edge computing cost with priority", "PSO", "GA", "NSGAII", "All")
    )

uploaded_file = st.file_uploader("Choose a file")
data = Load(uploaded_file)

if data is not None:
    random_lat, random_lon = Map(data)

    distances = distance_vectorized(random_lat, random_lon, data['latitude'].values, data['longitude'].values)

    num_rows = len(data)

    process_power = random_process_power(num_rows)
    task_sizes = random_task_sizes(num_rows)
    priorities = random_priority(num_rows)

    new_data = pd.DataFrame({
        'distance': distances,
        'processing_power': process_power,
        'task_size': task_sizes,
        'priority': priorities
    })

    data = pd.concat([data, new_data], axis=1)

    st.write(data)

    num_jobs = 500
    num_edge_devices = len(data['longitude'])

    random_allocation = np.random.randint(0, num_edge_devices, size=num_jobs)
    distances = np.tile(data['distance'].values, (num_jobs, 1))

    if option == "Edge computing cost with priority":
        random_cost = edge_computing_cost_with_priority(random_allocation, distances, data['processing_power'],
                                                        data['task_size'], data['priority'])
        st.write("\nRandom Allocation Cost:", random_cost)
    elif option == "PSO":
        pso = PSO(
            cost_func=lambda pos: edge_computing_cost_with_priority(np.round(pos).astype(int), distances,
                                                                    data['processing_power'],
                                                                    data['task_size'], data['priority']),
            num_particles=30, num_iterations=50, dim=num_jobs, bounds=[0, num_edge_devices - 1])

        best_pso_position, best_pso_cost = pso.optimize()

        st.write("\nBest PSO Cost:", best_pso_cost)
    elif option == "GA":
        best_ga_position, best_ga_cost = genetic_algorithm(num_jobs=num_jobs, num_edge_devices=num_edge_devices,
                                                           distances=distances,
                                                           processing_powers=data['processing_power'],
                                                           task_sizes=data['task_size'], priorities=data['priority'],
                                                           pop_size=50, num_generations=50, crossover_rate=0.8,
                                                           mutation_rate=0.02)

        st.write("\nBest GA Cost:", best_ga_cost)
    elif option == "NSGAII":
        nsga2 = NSGAII(pop_size=50, num_generations=50, distances=distances, processing_powers=data['processing_power'],
                       task_sizes=data['task_size'], priorities=data['priority'], num_jobs=num_jobs,
                       num_edge_devices=num_edge_devices)

        best_nsga2_population, best_nsga2_costs = nsga2.run()

        st.write("\nNSGA2:", min(best_nsga2_costs[:, 0]))

    elif option == "All":
        result_queue = Queue()
        threads = [
            threading.Thread(target=run_edge_computing_cost_with_priority, args=(result_queue, random_allocation,
                                                                                 distances,
                                                                                 data['processing_power'],
                                                                                 data['task_size'], data['priority'])),
            threading.Thread(target=run_PSO,
                             args=(result_queue,
                                   lambda pos: edge_computing_cost_with_priority(np.round(pos).astype(int), distances,
                                                                                 data['processing_power'],
                                                                                 data['task_size'], data['priority']),
                                   30, 50, num_jobs, [0, num_edge_devices - 1])),
            threading.Thread(target=run_genetic_algorithm,
                             args=(result_queue, num_jobs, num_edge_devices, distances, data['processing_power'],
                                   data['task_size'], data['priority'], 50, 50, 0.8, 0.02)),
            threading.Thread(target=run_NSGAII,
                             args=(result_queue, distances, data['processing_power'], data['task_size'],
                                   data['priority'], num_jobs,
                                   num_edge_devices, 50, 50))
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        while not result_queue.empty():
            algorithm, result = result_queue.get()
            st.write(f"{algorithm}: {result}")
