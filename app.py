import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time, json, random, os, io, base64
from datetime import datetime
from PIL import Image
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Import algorithms from separate modules
from algorithms.dijkstra import dijkstra_algorithm
from algorithms.astar import astar_algorithm
from algorithms.bellman_ford import bellman_ford_algorithm

# Page configuration and simplified CSS
st.set_page_config(page_title="Smart Traffic Flow Optimizer", page_icon="🚦", layout="wide")

# Simplified CSS
CUSTOM_CSS = """
<style>
  .main-header { font-size: 2.5rem; color: #FF6B35; font-weight: 700; }
  .sub-header { font-size: 1.5rem; color: #424242; font-weight: 500; }
  .card { padding: 20px; border-radius: 10px; background-color: #f8f9fa; 
          box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border-top: 4px solid #FF6B35; }
  .metric-card { background-color: #fff8f3; border-left: 5px solid #FF6B35; }
  .info-text { font-size: 0.9rem; color: #616161; }
  .highlight { background-color: #fff0e6; padding: 2px 5px; border-radius: 3px; font-weight: 500; color: #FF6B35; }
  .traffic-badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
  .traffic-low { background-color: #DCEDC8; color: #33691E; }
  .traffic-medium { background-color: #FFE0B2; color: #E65100; }
  .traffic-high { background-color: #FFCDD2; color: #B71C1C; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f5f5f5; border-radius: 4px 4px 0 0; padding: 10px; }
  .stTabs [aria-selected="true"] { background-color: #FF6B35 !important; color: white !important; }
  .stButton>button { background-color: #FF6B35; color: white; border: none; border-radius: 4px; padding: 0.5rem 1rem; }
  .stButton>button:hover { background-color: #E55A24; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Data loading and graph creation
@st.cache_data
def load_sample_data():
    if os.path.exists("data/city_graph.json"):
        with open("data/city_graph.json", "r") as f:
            return json.load(f)
    else:
        return generate_sample_city_data()

def generate_sample_city_data():
    """Generate a sample city graph with intersections and roads"""
    intersections = {
        "A": {"pos": (0, 0), "name": "Delhi"},
        "B": {"pos": (2, 4), "name": "Gurgaon"},
        "C": {"pos": (5, 2), "name": "Noida"},
        "D": {"pos": (3, -2), "name": "Faridabad"},
        "E": {"pos": (-3, 3), "name": "Sonipat"},
        "F": {"pos": (1, 6), "name": "Rohtak"},
        "G": {"pos": (6, 0), "name": "Greater Noida"},
        "H": {"pos": (-2, -3), "name": "Rewari"},
        "I": {"pos": (4, 5), "name": "Meerut"},
        "J": {"pos": (-4, -1), "name": "Jhajjar"}
    }
    
    roads = [
        {"from": "A", "to": "B", "distance": 30, "traffic": 0.8, "name": "NH-48"},
        {"from": "A", "to": "C", "distance": 25, "traffic": 0.5, "name": "DND Flyway"},
        {"from": "A", "to": "D", "distance": 28, "traffic": 0.3, "name": "Mathura Road"},
        {"from": "A", "to": "E", "distance": 45, "traffic": 0.6, "name": "GT Karnal Road"},
        {"from": "B", "to": "F", "distance": 70, "traffic": 0.2, "name": "NH-9"},
        {"from": "B", "to": "I", "distance": 80, "traffic": 0.4, "name": "KMP Expressway"},
        {"from": "C", "to": "G", "distance": 20, "traffic": 0.1, "name": "Noida-Greater Noida Expressway"},
        {"from": "C", "to": "I", "distance": 65, "traffic": 0.7, "name": "NH-58"},
        {"from": "D", "to": "G", "distance": 35, "traffic": 0.5, "name": "Yamuna Expressway"},
        {"from": "D", "to": "H", "distance": 90, "traffic": 0.3, "name": "KMP Expressway"},
        {"from": "E", "to": "F", "distance": 50, "traffic": 0.4, "name": "NH-9"},
        {"from": "E", "to": "J", "distance": 60, "traffic": 0.2, "name": "SH-20"},
        {"from": "F", "to": "I", "distance": 90, "traffic": 0.3, "name": "NH-334"},
        {"from": "G", "to": "I", "distance": 75, "traffic": 0.6, "name": "Eastern Peripheral Expressway"},
        {"from": "H", "to": "J", "distance": 55, "traffic": 0.4, "name": "SH-15"},
        {"from": "I", "to": "F", "distance": 90, "traffic": 0.2, "name": "NH-334B"},
        {"from": "J", "to": "H", "distance": 55, "traffic": 0.3, "name": "KMP Expressway"}
    ]
    
    return {"intersections": intersections, "roads": roads}

def create_graph_from_data(data, consider_traffic=True):
    """Create a NetworkX graph from the data"""
    G = nx.DiGraph()
    
    for node_id, node_data in data["intersections"].items():
        G.add_node(node_id, pos=node_data["pos"], name=node_data["name"])
    
    for road in data["roads"]:
        weight = road["distance"] * (1 + road["traffic"] * 2) if consider_traffic else road["distance"]
        G.add_edge(road["from"], road["to"], weight=weight, distance=road["distance"], 
                  traffic=road["traffic"], name=road["name"], color='blue', width=2)
    
    return G

def visualize_graph(G, path=None, title="NCR Traffic Network"):
    """Visualize the graph with optional path highlighting"""
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    edge_colors = ['#4CAF50' if G[u][v]['traffic'] < 0.3 else '#FF9800' if G[u][v]['traffic'] < 0.7 else '#F44336' for u, v in G.edges()]
    edge_widths = [2 + G[u][v]['traffic'] * 3 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='#FF6B35', alpha=0.9, edgecolors='white', linewidths=2)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrowsize=15, alpha=0.7)
    
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=5, edge_color='#3F51B5', arrowsize=20, alpha=1.0)
    
    node_labels = {node: f"{G.nodes[node]['name']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, font_weight='bold', font_color='white')
    
    edge_labels = {(u, v): f"{G[u][v]['name']}\n{G[u][v]['distance']}km, {'🟢' if G[u][v]['traffic'] < 0.3 else '🟠' if G[u][v]['traffic'] < 0.7 else '🔴'} {int(G[u][v]['traffic']*100)}%" 
                  for u, v in G.edges()}
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='#333333')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.axis('off')
    plt.tight_layout()
    return plt

def get_traffic_badge(traffic_level):
    """Return HTML for a traffic badge based on level"""
    level = int(traffic_level * 100)
    if level < 30:
        return f'<span class="traffic-badge traffic-low">{level}%</span>'
    elif level < 70:
        return f'<span class="traffic-badge traffic-medium">{level}%</span>'
    else:
        return f'<span class="traffic-badge traffic-high">{level}%</span>'

def simulate_traffic_change():
    """Simulate traffic changes over time"""
    data = load_sample_data()
    current_hour = datetime.now().hour
    is_rush_hour = (8 <= current_hour <= 10) or (17 <= current_hour <= 19)
    
    for road in data["roads"]:
        change = random.uniform(-0.15, 0.15)
        if is_rush_hour:
            if road["to"] in ["A", "B", "C"] or road["from"] in ["A", "B", "C"]:
                change += 0.2
        road["traffic"] = max(0.1, min(0.95, road["traffic"] + change))
    
    return data

def create_map_visualization(G, path=None, map_type="folium"):
    """Create map visualization based on type"""
    if map_type == "folium":
        base_lat, base_lon = 28.6139, 77.2090  # Delhi coordinates
        scale = 0.05  # Scale factor for visualization
        m = folium.Map(location=[base_lat, base_lon], zoom_start=9, tiles="CartoDB positron")
        pos = nx.get_node_attributes(G, 'pos')
        
        for node, position in pos.items():
            lat, lon = base_lat + position[0] * scale, base_lon + position[1] * scale
            is_path_node = path and node in path
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{G.nodes[node]['name']}</b>",
                icon=folium.Icon(color='red' if is_path_node else 'blue', icon='info-sign')
            ).add_to(m)
        
        for u, v, data in G.edges(data=True):
            u_lat, u_lon = base_lat + pos[u][0] * scale, base_lon + pos[u][1] * scale
            v_lat, v_lon = base_lat + pos[v][0] * scale, base_lon + pos[v][1] * scale
            
            traffic = data['traffic']
            color = 'green' if traffic < 0.3 else 'orange' if traffic < 0.7 else 'red'
            weight = 3 if traffic < 0.3 else 4 if traffic < 0.7 else 5
            
            is_path_edge = path and u in path and v in path and path.index(u) == path.index(v) - 1
            if is_path_edge:
                color, weight = 'purple', 6
            
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                popup=f"<b>{data['name']}</b><br>Distance: {data['distance']} km<br>Traffic: {int(traffic*100)}%",
                color=color, weight=weight, opacity=0.8
            ).add_to(m)
        
        return m
    elif map_type == "plotly":
        pos = nx.get_node_attributes(G, 'pos')
        node_x, node_y = zip(*[pos[node] for node in G.nodes()])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=[G.nodes[node]['name'] for node in G.nodes()],
            textposition="top center", textfont=dict(size=10),
            marker=dict(color='#FF6B35', size=20, line_width=2, line=dict(color='white')),
            hovertext=[f"City: {G.nodes[node]['name']}" for node in G.nodes()]
        )
        
        edge_traces = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            traffic = G[u][v]['traffic']
            
            color = 'rgba(76, 175, 80, 0.7)' if traffic < 0.3 else 'rgba(255, 152, 0, 0.7)' if traffic < 0.7 else 'rgba(244, 67, 54, 0.7)'
            width = 2 + traffic * 3
            
            is_path_edge = path and u in path and v in path and path.index(u) == path.index(v) - 1
            if is_path_edge:
                color, width = 'rgba(63, 81, 181, 1.0)', 5
            
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=width, color=color),
                hoverinfo='text', mode='lines',
                text=f"Road: {G[u][v]['name']}<br>Distance: {G[u][v]['distance']} km<br>Traffic: {int(traffic*100)}%"
            ))
        
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='NCR Traffic Network', showlegend=False, hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(248,249,250,1)', paper_bgcolor='rgba(248,249,250,1)'
            )
        )
        
        return fig

def main():
    # Sidebar
    st.sidebar.markdown('<p class="main-header">🚦 Smart Traffic</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Flow Optimizer</p>', unsafe_allow_html=True)
    
    # Time indicator
    current_time = datetime.now().strftime("%H:%M")
    current_hour = datetime.now().hour
    time_icon = "🌅" if 5 <= current_hour < 12 else "☀️" if 12 <= current_hour < 17 else "🌆" if 17 <= current_hour < 21 else "🌙"
    time_greeting = "Good Morning" if 5 <= current_hour < 12 else "Good Afternoon" if 12 <= current_hour < 17 else "Good Evening" if 17 <= current_hour < 21 else "Good Night"
    
    st.sidebar.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <div style="font-size: 0.8rem; color: #666;">Current Time</div>
        <div style="font-size: 1.2rem; font-weight: bold;">{time_icon} {current_time}</div>
        <div style="font-size: 0.9rem;">{time_greeting}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # App tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚗 Route Optimizer", "🔄 Traffic Simulation", "📊 Algorithm Comparison", "ℹ️ About"])
    
    # Tab 1: Route Optimizer
    with tab1:
        st.markdown('<p class="main-header">🚗 Smart Route Optimizer</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Load data and create graph
            data = load_sample_data()
            consider_traffic = st.checkbox("Consider Traffic Conditions", value=True)
            G = create_graph_from_data(data, consider_traffic)
            
            # Source and destination selection
            nodes = list(data["intersections"].keys())
            node_names = [f"{data['intersections'][node]['name']} ({node})" for node in nodes]
            
            source = st.selectbox("Select Starting Point", node_names, index=0)
            destination = st.selectbox("Select Destination", node_names, index=len(node_names)-1)
            
            source_node = source.split("(")[1].split(")")[0].strip()
            dest_node = destination.split("(")[1].split(")")[0].strip()
            
            # Algorithm selection
            algorithm = st.selectbox(
                "Select Routing Algorithm",
                ["Dijkstra's Algorithm", "A* Algorithm", "Bellman-Ford Algorithm"]
            )
            
            if st.button("Calculate Optimal Route"):
                with st.spinner("Calculating optimal route..."):
                    start_time = time.time()
                    
                    # Run selected algorithm
                    if algorithm == "Dijkstra's Algorithm":
                        distance, path = dijkstra_algorithm(G, source_node, dest_node)
                    elif algorithm == "A* Algorithm":
                        distance, path = astar_algorithm(G, source_node, dest_node)
                    else:  # Bellman-Ford
                        distance, path = bellman_ford_algorithm(G, source_node, dest_node)
                    
                    computation_time = time.time() - start_time
                    
                    if path:
                        # Calculate metrics
                        total_distance = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
                        total_traffic = sum(G[path[i]][path[i+1]]['traffic'] for i in range(len(path)-1))
                        avg_traffic = total_traffic / (len(path)-1)
                        avg_speed = 60 * (1 - avg_traffic * 0.7)  # km/h
                        travel_time = (total_distance / avg_speed) * 60  # minutes
                        
                        # Display metrics
                        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### Route Summary")
                        
                        path_cities = [f"{G.nodes[node]['name']}" for node in path]
                        st.markdown(f"**Route:** {' → '.join(path_cities)}")
                        
                        col1a, col2a, col3a = st.columns(3)
                        with col1a:
                            st.metric("Total Distance", f"{total_distance:.1f} km")
                        with col2a:
                            st.markdown(f"**Traffic Level:**<br>{get_traffic_badge(avg_traffic)}", unsafe_allow_html=True)
                        with col3a:
                            st.metric("Est. Travel Time", f"{travel_time:.0f} min")
                        
                        st.markdown(f"**Computation Time:** {computation_time*1000:.2f} ms")
                        
                        # Display directions
                        st.markdown("### Turn-by-Turn Directions")
                        for i in range(len(path)-1):
                            from_city = G.nodes[path[i]]['name']
                            to_city = G.nodes[path[i+1]]['name']
                            road_name = G[path[i]][path[i+1]]['name']
                            distance = G[path[i]][path[i+1]]['distance']
                            traffic = G[path[i]][path[i+1]]['traffic']
                            
                            st.markdown(f"{i+1}. Take <span class='highlight'>{road_name}</span> from {from_city} to {to_city} ({distance} km) {get_traffic_badge(traffic)}", unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"No path found between {source_node} and {dest_node}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### NCR Traffic Network")
            
            # Create tabs for different visualizations
            map_tab1, map_tab2 = st.tabs(["Network Graph", "Interactive Map"])
            
            with map_tab1:
                # Visualize the graph
                fig = visualize_graph(G, path=path if 'path' in locals() else None)
                st.pyplot(fig)
            
            with map_tab2:
                # Create interactive map
                m = create_map_visualization(G, path=path if 'path' in locals() else None, map_type="folium")
                folium_static(m)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Traffic Simulation
    with tab2:
        st.markdown('<p class="main-header">🔄 Traffic Simulation</p>', unsafe_allow_html=True)
        
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
            
            # Display traffic table
            st.markdown("### Current Traffic Conditions")
            data_to_display = st.session_state.get('simulation_data', load_sample_data())
            
            # Create a more visual traffic table
            roads_df = pd.DataFrame(data_to_display["roads"])
            roads_df['from_city'] = roads_df['from'].apply(lambda x: data_to_display["intersections"][x]["name"])
            roads_df['to_city'] = roads_df['to'].apply(lambda x: data_to_display["intersections"][x]["name"])
            roads_df['traffic_html'] = roads_df['traffic'].apply(get_traffic_badge)
            
            styled_df = pd.DataFrame({
                'Road': roads_df['name'],
                'From': roads_df['from_city'],
                'To': roads_df['to_city'],
                'Distance': roads_df['distance'],
                'Traffic': roads_df['traffic_html']
            })
            
            st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Traffic Visualization")
            
            # Create graph from current data
            current_data = st.session_state.get('simulation_data', load_sample_data())
            G = create_graph_from_data(current_data, consider_traffic=True)
            
            # Visualize
            fig = visualize_graph(G, title="Current NCR Traffic Conditions")
            st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Algorithm Comparison
    with tab3:
        st.markdown('<p class="main-header">📊 Algorithm Comparison</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Load data and create graph
        data = load_sample_data()
        G = create_graph_from_data(data, consider_traffic=True)
        
        # Source and destination selection
        nodes = list(data["intersections"].keys())
        node_names = [f"{data['intersections'][node]['name']} ({node})" for node in nodes]
        
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Select Starting Point", node_names, index=0, key="comp_source")
        with col2:
            destination = st.selectbox("Select Destination", node_names, index=len(node_names)-1, key="comp_dest")
        
        source_node = source.split("(")[1].split(")")[0].strip()
        dest_node = destination.split("(")[1].split(")")[0].strip()
        
        if st.button("Compare Algorithms"):
            with st.spinner("Running comparison..."):
                results = []
                
                # Run all algorithms and time them
                start_time = time.time()
                dijkstra_dist, dijkstra_path = dijkstra_algorithm(G, source_node, dest_node)
                dijkstra_time = time.time() - start_time
                
                start_time = time.time()
                astar_dist, astar_path = astar_algorithm(G, source_node, dest_node)
                astar_time = time.time() - start_time
                
                start_time = time.time()
                bellman_ford_dist, bellman_ford_path = bellman_ford_algorithm(G, source_node, dest_node)
                bellman_ford_time = time.time() - start_time
                
                # Format results
                def format_path(path):
                    return "No path found" if not path else " → ".join([f"{data['intersections'][node]['name']}" for node in path])
                
                # Collect results
                results.append({
                    "Algorithm": "Dijkstra's Algorithm",
                    "Path": format_path(dijkstra_path),
                    "Distance (km)": f"{dijkstra_dist:.2f}" if dijkstra_path else "N/A",
                    "Computation Time (ms)": f"{dijkstra_time*1000:.2f}"
                })
                
                results.append({
                    "Algorithm": "A* Algorithm",
                    "Path": format_path(astar_path),
                    "Distance (km)": f"{astar_dist:.2f}" if astar_path else "N/A",
                    "Computation Time (ms)": f"{astar_time*1000:.2f}"
                })
                
                results.append({
                    "Algorithm": "Bellman-Ford Algorithm",
                    "Path": format_path(bellman_ford_path),
                    "Distance (km)": f"{bellman_ford_dist:.2f}" if bellman_ford_path else "N/A",
                    "Computation Time (ms)": f"{bellman_ford_time*1000:.2f}"
                })
                
                # Display results
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                
                # Create bar chart
                fig = px.bar(
                    pd.DataFrame(results), 
                    x="Algorithm", 
                    y=[float(t.split()[0]) for t in pd.DataFrame(results)["Computation Time (ms)"]],
                    labels={"y": "Computation Time (ms)"},
                    title="Algorithm Performance Comparison",
                    color="Algorithm",
                    color_discrete_sequence=["#FF6B35", "#4CAF50", "#3F51B5"]
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: About
    with tab4:
        st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### Smart Traffic Flow Optimization System for NCR
        
        This application demonstrates the use of graph algorithms for optimizing traffic flow in the National Capital Region (NCR) of India.
        It implements several key algorithms:
        
        - **Dijkstra's Algorithm**: A greedy algorithm that finds the shortest path between nodes in a graph
        - **A* Algorithm**: An extension of Dijkstra's that uses heuristics to speed up the search
        - **Bellman-Ford Algorithm**: An algorithm that computes shortest paths from a single source vertex to all other vertices
        
        ### Technologies Used
        
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **NetworkX**: Graph manipulation and analysis
        - **Matplotlib & Plotly**: Data visualization
        - **Folium**: Interactive maps
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown(
        """
        <div style='text-align: center; color: #3D52A0; font-size: 0.9rem; margin-top: 2rem;'>
            🚦 <strong>Smart Traffic Flow Optimization System</strong><br>
            Developed with DAA using Python & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()