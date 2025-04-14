import heapq

def dijkstra_algorithm(G, source, target):
    
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('infinity') for node in G.nodes()}
    distances[source] = 0
    
    # Initialize predecessors dictionary to reconstruct the path
    predecessors = {node: None for node in G.nodes()}
    
    # Priority queue for nodes to visit
    # Format: (distance, node)
    priority_queue = [(0, source)]
    
    # Set to keep track of visited nodes
    visited = set()
    
    while priority_queue:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If we reached the target, we can stop
        if current_node == target:
            break
        
        # Skip if we've already processed this node
        if current_node in visited:
            continue
        
        # Mark as visited
        visited.add(current_node)
        
        # Check all neighbors
        for neighbor in G.neighbors(current_node):
            # Skip if already visited
            if neighbor in visited:
                continue
            
            # Calculate new distance
            weight = G[current_node][neighbor]['weight']
            distance = current_distance + weight
            
            # If we found a better path, update
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # Reconstruct the path
    if target not in visited and distances[target] == float('infinity'):
        return float('infinity'), []
    
    path = []
    current = target
    while current:
        path.append(current)
        current = predecessors[current]
    
    # Reverse to get path from source to target
    path.reverse()
    
    return distances[target], path
