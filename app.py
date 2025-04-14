import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from algorithms.dijkstra import dijkstra_algorithm
from algorithms.astar import astar_algorithm
from algorithms.bellman_ford import bellman_ford_algorithm
import json
from streamlit_folium import folium_static
import folium
import random
import os
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Smart Traffic Flow Optimizer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f1f8fe;
        border-left: 5px solid #1E88E5;
    }
    .info-text {
        font-size: 0.9rem;
        color: #616161;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_sample_data():
    # Check if data file exists, otherwise create sample data
    if os.path.exists("data/city_graph.json"):
        with open("data/city_graph.json", "r") as f:
            return json.load(f)
    else:
        # Create sample data
        return generate_sample_city_data()

def generate_sample_city_data():
    """Generate a sample city graph with intersections and roads"""
    intersections = {
        "A": {"pos": (0, 0), "name": "Downtown"},
        "B": {"pos": (2, 4), "name": "Uptown"},
        "C": {"pos": (5, 2), "name": "Eastside"},
        "D": {"pos": (3, -2), "name": "Southside"},
        "E": {"pos": (-3, 3), "name": "Westside"},
        "F": {"pos": (1, 6), "name": "North End"},
        "G": {"pos": (6, 0), "name": "East End"},
        "H": {"pos": (-2, -3), "name": "Southwest"},
        "I": {"pos": (4, 5), "name": "Northeast"},
        "J": {"pos": (-4, -1), "name": "West End"}
    }
    
    roads = [
        {"from": "A", "to": "B", "distance": 5, "traffic": 0.8, "name": "Main St"},
        {"from": "A", "to": "C", "distance": 6, "traffic": 0.5, "name": "Broadway"},
        {"from": "A", "to": "D", "distance": 4, "traffic": 0.3, "name": "Park Ave"},
        {"from": "A", "to": "E", "distance": 7, "traffic": 0.6, "name": "1st Ave"},
        {"from": "B", "to": "F", "distance": 3, "traffic": 0.2, "name": "Hill Rd"},
        {"from": "B", "to": "I", "distance": 4, "traffic": 0.4, "name": "University Blvd"},
        {"from": "C", "to": "G", "distance": 2, "traffic": 0.1, "name": "Market St"},
        {"from": "C", "to": "I", "distance": 5, "traffic": 0.7, "name": "Highland Dr"},
        {"from": "D", "to": "G", "distance": 6, "traffic": 0.5, "name": "River Rd"},
        {"from": "D", "to": "H", "distance": 5, "traffic": 0.3, "name": "Valley Blvd"},
        {"from": "E", "to": "F", "distance": 4, "traffic": 0.4, "name": "Forest Ave"},
        {"from": "E", "to": "J", "distance": 3, "traffic": 0.2, "name": "Lake Dr"},
        {"from": "F", "to": "I", "distance": 3, "traffic": 0.3, "name": "Mountain View Rd"},
        {"from": "G", "to": "I", "distance": 7, "traffic": 0.6, "name": "Sunset Blvd"},
        {"from": "H", "to": "J", "distance": 4, "traffic": 0.4, "name": "Ocean Dr"},
        {"from": "I", "to": "F", "distance": 2, "traffic": 0.2, "name": "Ridge Rd"},
        {"from": "J", "to": "H", "distance": 3, "traffic": 0.3, "name": "Bay St"}
    ]
    
    return {"intersections": intersections, "roads": roads}

def create_graph_from_data(data, consider_traffic=True):
    """Create a NetworkX graph from the data"""
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in data["intersections"].items():
        G.add_node(node_id, pos=node_data["pos"], name=node_data["name"])
    
    # Add edges
    for road in data["roads"]:
        from_node = road["from"]
        to_node = road["to"]
        distance = road["distance"]
        traffic = road["traffic"]
        name = road["name"]
        
        # Calculate weight based on distance and traffic
        weight = distance * (1 + traffic * 2) if consider_traffic else distance
        
        G.add_edge(from_node, to_node, weight=weight, distance=distance, 
                  traffic=traffic, name=name, color='blue', width=2)
    
    return G

def visualize_graph(G, path=None, title="City Traffic Network"):
    """Visualize the graph with optional path highlighting"""
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
    
    # Draw edges
    edge_colors = [G[u][v].get('color', 'blue') for u, v in G.edges()]
    edge_widths = [G[u][v].get('width', 2) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                          arrowsize=15, connectionstyle='arc3,rad=0.1')
    
    # Highlight path if provided
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, 
                              edge_color='red', arrowsize=20)
    
    # Draw labels
    node_labels = {node: f"{node}: {G.nodes[node]['name']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
    
    # Draw edge labels
    edge_labels = {(u, v): f"{G[u][v]['name']}\n{G[u][v]['distance']}km, {int(G[u][v]['traffic']*100)}% traffic" 
                  for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    return plt

def simulate_traffic_change():
    """Simulate traffic changes over time"""
    data = load_sample_data()
    for road in data["roads"]:
        # Randomly adjust traffic levels (between -0.2 and +0.2)
        change = random.uniform(-0.2, 0.2)
        road["traffic"] = max(0.1, min(1.0, road["traffic"] + change))
    return data

def create_folium_map(G, path=None):
    """Create an interactive folium map"""
    # Create a base map centered on the average position of nodes
    pos = nx.get_node_attributes(G, 'pos')
    avg_lat = sum(p[0] for p in pos.values()) / len(pos)
    avg_lon = sum(p[1] for p in pos.values()) / len(pos)
    
    # Scale positions to realistic lat/lon values (just for visualization)
    base_lat, base_lon = 40.7128, -74.0060  # New York City coordinates
    scale = 0.01  # Scale factor
    
    m = folium.Map(location=[base_lat, base_lon], zoom_start=13)
    
    # Add nodes as markers
    for node, position in pos.items():
        lat = base_lat + position[0] * scale
        lon = base_lon + position[1] * scale
        
        popup_text = f"<b>{node}: {G.nodes[node]['name']}</b>"
        
        # Use different marker for nodes in the path
        if path and node in path:
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        else:
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
    
    # Add edges as lines
    for u, v, data in G.edges(data=True):
        u_pos = pos[u]
        v_pos = pos[v]
        
        u_lat = base_lat + u_pos[0] * scale
        u_lon = base_lon + u_pos[1] * scale
        v_lat = base_lat + v_pos[0] * scale
        v_lon = base_lon + v_pos[1] * scale
        
        # Determine line color and weight based on traffic
        traffic = data['traffic']
        if traffic < 0.3:
            color = 'green'
        elif traffic < 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        weight = 2 + traffic * 5
        
        # Highlight path edges
        if path and u in path and v in path and path.index(u) == path.index(v) - 1:
            color = 'purple'
            weight = 8
        
        popup_text = f"<b>{data['name']}</b><br>Distance: {data['distance']}km<br>Traffic: {int(traffic*100)}%"
        
        folium.PolyLine(
            locations=[[u_lat, u_lon], [v_lat, v_lon]],
            popup=popup_text,
            color=color,
            weight=weight,
            opacity=0.8
        ).add_to(m)
    
    return m

def main():
    # Sidebar
    st.sidebar.markdown('<p class="main-header">🚦 Smart Traffic</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Flow Optimizer</p>', unsafe_allow_html=True)
    
    # App sections
    tab1, tab2, tab3, tab4 = st.tabs(["Route Optimizer", "Traffic Simulation", "Algorithm Comparison", "About"])
    
    with tab1:
        st.markdown('<p class="main-header">🚗 Smart Route Optimizer</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Find the optimal route between two locations considering current traffic conditions.
        The system uses advanced graph algorithms to calculate the most efficient path.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Load data and create graph
            data = load_sample_data()
            consider_traffic = st.checkbox("Consider Traffic Conditions", value=True)
            G = create_graph_from_data(data, consider_traffic)
            
            # Source and destination selection
            nodes = list(data["intersections"].keys())
            node_names = [f"{node}: {data['intersections'][node]['name']}" for node in nodes]
            
            source = st.selectbox("Select Starting Point", node_names, index=0)
            destination = st.selectbox("Select Destination", node_names, index=len(node_names)-1)
            
            source_node = source.split(":")[0].strip()
            dest_node = destination.split(":")[0].strip()
            
            # Algorithm selection
            algorithm = st.selectbox(
                "Select Routing Algorithm",
                ["Dijkstra's Algorithm", "A* Algorithm", "Bellman-Ford Algorithm"]
            )
            
            if st.button("Calculate Optimal Route"):
                with st.spinner("Calculating optimal route..."):
                    start_time = time.time()
                    
                    if algorithm == "Dijkstra's Algorithm":
                        distance, path = dijkstra_algorithm(G, source_node, dest_node)
                    elif algorithm == "A* Algorithm":
                        distance, path = astar_algorithm(G, source_node, dest_node)
                    else:  # Bellman-Ford
                        distance, path = bellman_ford_algorithm(G, source_node, dest_node)
                    
                    computation_time = time.time() - start_time
                    
                    if path:
                        # Highlight the path in the graph
                        for u, v in zip(path[:-1], path[1:]):
                            G[u][v]['color'] = 'red'
                            G[u][v]['width'] = 4
                        
                        # Calculate metrics
                        total_distance = 0
                        total_traffic_score = 0
                        road_names = []
                        
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            total_distance += G[u][v]['distance']
                            total_traffic_score += G[u][v]['traffic']
                            road_names.append(G[u][v]['name'])
                        
                        avg_traffic = total_traffic_score / (len(path)-1) if len(path) > 1 else 0
                        
                        # Display metrics
                        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### Route Summary")
                        st.markdown(f"**Path:** {' → '.join(path)}")
                        st.markdown(f"**Total Distance:** {total_distance:.1f} km")
                        st.markdown(f"**Average Traffic Level:** {avg_traffic:.2f} (0-1 scale)")
                        st.markdown(f"**Computation Time:** {computation_time*1000:.2f} ms")
                        
                        # Display route directions
                        st.markdown("### Turn-by-Turn Directions")
                        for i, road in enumerate(road_names):
                            st.markdown(f"{i+1}. Take <span class='highlight'>{road}</span> from {path[i]} to {path[i+1]}", unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"No path found between {source_node} and {dest_node}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### City Traffic Network")
            
            # Create tabs for different visualizations
            map_tab1, map_tab2 = st.tabs(["Network Graph", "Interactive Map"])
            
            with map_tab1:
                # Visualize the graph
                fig = visualize_graph(G, path=path if 'path' in locals() else None)
                st.pyplot(fig)
            
            with map_tab2:
                # Create interactive map
                m = create_folium_map(G, path=path if 'path' in locals() else None)
                folium_static(m)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<p class="main-header">🔄 Traffic Simulation</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Simulate changing traffic conditions and observe how they affect optimal routes.
        This helps understand the dynamic nature of urban traffic patterns.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Simulation Controls")
            
            if st.button("Simulate Traffic Change"):
                st.session_state.simulation_data = simulate_traffic_change()
                st.success("Traffic conditions updated!")
            
            if st.button("Reset to Default"):
                if 'simulation_data' in st.session_state:
                    del st.session_state.simulation_data
                st.success("Traffic conditions reset to default!")
            
            # Display traffic conditions table
            st.markdown("### Current Traffic Conditions")
            
            data_to_display = st.session_state.get('simulation_data', load_sample_data())
            
            roads_df = pd.DataFrame(data_to_display["roads"])
            roads_df['traffic'] = roads_df['traffic'].apply(lambda x: f"{int(x*100)}%")
            roads_df = roads_df.rename(columns={
                'from': 'From', 'to': 'To', 'distance': 'Distance (km)',
                'traffic': 'Traffic Level', 'name': 'Road Name'
            })
            
            st.dataframe(roads_df[['Road Name', 'From', 'To', 'Distance (km)', 'Traffic Level']], 
                        use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Traffic Visualization")
            
            # Create graph from current data
            current_data = st.session_state.get('simulation_data', load_sample_data())
            G = create_graph_from_data(current_data, consider_traffic=True)
            
            # Visualize
            fig = visualize_graph(G, title="Current Traffic Conditions")
            st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<p class="main-header">📊 Algorithm Comparison</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Compare different routing algorithms to understand their performance characteristics.
        This helps in selecting the most appropriate algorithm for specific traffic scenarios.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Load data and create graph
        data = load_sample_data()
        G = create_graph_from_data(data, consider_traffic=True)
        
        # Source and destination selection
        nodes = list(data["intersections"].keys())
        node_names = [f"{node}: {data['intersections'][node]['name']}" for node in nodes]
        
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Select Starting Point", node_names, index=0, key="comp_source")
        with col2:
            destination = st.selectbox("Select Destination", node_names, index=len(node_names)-1, key="comp_dest")
        
        source_node = source.split(":")[0].strip()
        dest_node = destination.split(":")[0].strip()
        
        if st.button("Compare Algorithms"):
            with st.spinner("Running comparison..."):
                results = []
                
                # Run Dijkstra's Algorithm
                start_time = time.time()
                dijkstra_dist, dijkstra_path = dijkstra_algorithm(G, source_node, dest_node)
                dijkstra_time = time.time() - start_time
                
                # Run A* Algorithm
                start_time = time.time()
                astar_dist, astar_path = astar_algorithm(G, source_node, dest_node)
                astar_time = time.time() - start_time
                
                # Run Bellman-Ford Algorithm
                start_time = time.time()
                bellman_ford_dist, bellman_ford_path = bellman_ford_algorithm(G, source_node, dest_node)
                bellman_ford_time = time.time() - start_time
                
                # Collect results
                results.append({
                    "Algorithm": "Dijkstra's Algorithm",
                    "Path": " → ".join(dijkstra_path) if dijkstra_path else "No path found",
                    "Distance": f"{dijkstra_dist:.2f}" if dijkstra_path else "N/A",
                    "Computation Time (ms)": f"{dijkstra_time*1000:.2f}",
                    "Path Length": len(dijkstra_path) - 1 if dijkstra_path else 0
                })
                
                results.append({
                    "Algorithm": "A* Algorithm",
                    "Path": " → ".join(astar_path) if astar_path else "No path found",
                    "Distance": f"{astar_dist:.2f}" if astar_path else "N/A",
                    "Computation Time (ms)": f"{astar_time*1000:.2f}",
                    "Path Length": len(astar_path) - 1 if astar_path else 0
                })
                
                results.append({
                    "Algorithm": "Bellman-Ford Algorithm",
                    "Path": " → ".join(bellman_ford_path) if bellman_ford_path else "No path found",
                    "Distance": f"{bellman_ford_dist:.2f}" if bellman_ford_path else "N/A",
                    "Computation Time (ms)": f"{bellman_ford_time*1000:.2f}",
                    "Path Length": len(bellman_ford_path) - 1 if bellman_ford_path else 0
                })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Create bar chart for computation time
                fig, ax = plt.subplots(figsize=(10, 5))
                algorithms = [r["Algorithm"] for r in results]
                times = [float(r["Computation Time (ms)"]) for r in results]
                
                bars = ax.bar(algorithms, times, color=['#1E88E5', '#FFC107', '#4CAF50'])
                ax.set_ylabel('Computation Time (ms)')
                ax.set_title('Algorithm Performance Comparison')
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.2f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Display algorithm characteristics
                st.markdown("### Algorithm Characteristics")
                
                characteristics = pd.DataFrame([
                    {
                        "Algorithm": "Dijkstra's Algorithm",
                        "Time Complexity": "O(V² + E)",
                        "Space Complexity": "O(V)",
                        "Handles Negative Weights": "No",
                        "Best Use Case": "Finding shortest paths in non-negative weighted graphs"
                    },
                    {
                        "Algorithm": "A* Algorithm",
                        "Time Complexity": "O(E)",
                        "Space Complexity": "O(V)",
                        "Handles Negative Weights": "No",
                        "Best Use Case": "Finding paths with heuristic information available"
                    },
                    {
                        "Algorithm": "Bellman-Ford Algorithm",
                        "Time Complexity": "O(V·E)",
                        "Space Complexity": "O(V)",
                        "Handles Negative Weights": "Yes",
                        "Best Use Case": "Detecting negative cycles and handling negative weights"
                    }
                ])
                
                st.dataframe(characteristics, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### Smart Traffic Flow Optimization System
        
        This application demonstrates the use of graph algorithms for optimizing traffic flow in urban environments.
        It implements several key algorithms covered in the course:
        
        - **Dijkstra's Algorithm**: A greedy algorithm that finds the shortest path between nodes in a graph
        - **A* Algorithm**: An extension of Dijkstra's that uses heuristics to speed up the search
        - **Bellman-Ford Algorithm**: An algorithm that computes shortest paths from a single source vertex to all other vertices
        
        ### Course Outcomes Addressed
        
        1. **CO1**: The application demonstrates asymptotic notations by analyzing algorithm complexity
        2. **CO2**: It implements various algorithm paradigms including greedy algorithms
        3. **CO5**: It applies Dijkstra's, Bellman-Ford, and other algorithms to solve real-world problems like traffic routing
        
        ### Technologies Used
        
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **NetworkX**: Graph manipulation and analysis
        - **Matplotlib & Folium**: Data visualization
        - **Pandas**: Data manipulation
        
        ### Future Enhancements
        
        - Integration with real-time traffic data APIs
        - Machine learning models to predict traffic patterns
        - More sophisticated traffic simulation models
        - Support for multi-modal transportation routing
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
