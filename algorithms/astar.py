import heapq
import math

def astar_algorithm(G, source, target):
    
    # Get positions for heuristic calculation
    pos = {node: data['pos'] for node, data in G.nodes(data=True)}
    
    # Heuristic function (Euclidean distance)
    def heuristic(node1, node2):
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Initialize g_score (cost from start to current node)
    g_score = {node: float('infinity') for node in G.nodes()}
    g_score[source] = 0
    
    # Initialize f_score (estimated total cost from start to goal through current node)
    f_score = {node: float('infinity') for node in G.nodes()}
    f_score[source] = heuristic(source, target)
    
    # Priority queue for nodes to visit
    # Format: (f_score, node)
    open_set = [(f_score[source], source)]
    
    # Set to keep track of nodes in the open set
    open_set_hash = {source}
    
    # Initialize predecessors dictionary to reconstruct the path
    predecessors = {node: None for node in G.nodes()}
    
    while open_set:
        # Get node with smallest f_score
        _, current_node = heapq.heappop(open_set)
        open_set_hash.remove(current_node)
        
        # If we reached the target, we can stop
        if current_node == target:
            break
        
        # Check all neighbors
        for neighbor in G.neighbors(current_node):
            # Calculate tentative g_score
            weight = G[current_node][neighbor]['weight']
            tentative_g_score = g_score[current_node] + weight
            
            # If we found a better path, update
            if tentative_g_score < g_score[neighbor]:
                # Update path and scores
                predecessors[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)
                
                # Add to open set if not already there
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
    
    # Reconstruct the path
    if target not in predecessors and predecessors[target] is None:
        return float('infinity'), []
    
    path = []
    current = target
    while current:
        path.append(current)
        current = predecessors[current]
    
    # Reverse to get path from source to target
    path.reverse()
    
    return g_score[target], path
