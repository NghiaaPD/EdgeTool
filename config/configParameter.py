import numpy as np


def distance_vectorized(random_lat, random_lon, latitudes, longitudes):
    R = 6371.0

    lat1 = np.radians(random_lat)
    lon1 = np.radians(random_lon)

    lat2 = np.radians(latitudes)
    lon2 = np.radians(longitudes)

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_km = R * c

    return distance_km


def random_process_power(num_edge_devices: int):
    processing_powers = np.random.uniform(low=10, high=100, size=num_edge_devices)
    return processing_powers


def random_task_sizes(num_jobs: int):
    task_sizes = np.random.uniform(low=100, high=1000, size=num_jobs)
    return task_sizes


def random_priority(num_jobs: int):
    priorities = np.random.uniform(low=1, high=10, size=num_jobs)
    return priorities
