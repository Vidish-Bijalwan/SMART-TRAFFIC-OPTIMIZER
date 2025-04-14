def bellman_ford_algorithm(G, source, target):

    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('infinity') for node in G.nodes()}
    distances[source] = 0
    
    # Initialize predecessors dictionary to reconstruct the path
    predecessors = {node: None for node in G.nodes()}
    
    # Get all edges
    edges = list(G.edges(data=True))
    
    # Relax edges |V| - 1 times
    for _ in range(len(G.nodes()) - 1):
        for u, v, data in edges:
            weight = data['weight']
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
    
    # Check for negative weight cycles
    for u, v, data in edges:
        weight = data['weight']
        if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
            # Negative weight cycle detected
            return float('infinity'), []
    
    # Reconstruct the path
    if distances[target] == float('infinity'):
        return float('infinity'), []
    
    path = []
    current = target
    while current:
        path.append(current)
        current = predecessors[current]
    
    # Reverse to get path from source to target
    path.reverse()
    
    return distances[target], path
