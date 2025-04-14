import networkx as nx
import random

def generate_random_graph(num_nodes=10, edge_probability=0.3, min_weight=1, max_weight=10):
    
    G = nx.DiGraph()
    
    # Add nodes with positions
    for i in range(num_nodes):
        # Generate random positions in a 2D grid
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        G.add_node(i, pos=(x, y), name=f"Node {i}")
    
    # Add edges with random weights
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < edge_probability:
                weight = random.uniform(min_weight, max_weight)
                traffic = random.random()  # Random traffic level between 0 and 1
                G.add_edge(i, j, weight=weight, distance=weight, traffic=traffic, 
                          name=f"Road {i}-{j}", color='blue', width=2)
    
    return G

def calculate_path_metrics(G, path):
    """
    Calculate metrics for a given path
    
    Args:
        G: NetworkX graph
        path: List of nodes representing a path
        
    Returns:
        dict: Dictionary containing path metrics
    """
    if not path or len(path) < 2:
        return {
            "distance": 0,
            "traffic_level": 0,
            "travel_time": 0,
            "num_intersections": 0
        }
    
    total_distance = 0
    total_traffic = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G[u][v]
        total_distance += edge_data['distance']
        total_traffic += edge_data['traffic']
    
    avg_traffic = total_traffic / (len(path) - 1)
    # Estimate travel time based on distance and traffic
    # Assuming base speed of 60 km/h with no traffic
    travel_time = total_distance / 60 * (1 + 2 * avg_traffic)  # in hours
    
    return {
        "distance": total_distance,
        "traffic_level": avg_traffic,
        "travel_time": travel_time * 60,  # convert to minutes
        "num_intersections": len(path) - 1
    }
