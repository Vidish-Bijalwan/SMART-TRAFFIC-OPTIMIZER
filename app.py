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
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import io
import base64
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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
        color: #FF6B35;
        font-weight: 700;
        margin-bottom: 0.5rem;
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
        border-top: 4px solid #FF6B35;
    }
    .metric-card {
        background-color: #fff8f3;
        border-left: 5px solid #FF6B35;
    }
    .info-text {
        font-size: 0.9rem;
        color: #616161;
    }
    .highlight {
        background-color: #fff0e6;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: 500;
        color: #FF6B35;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f5f5f5;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B35 !important;
        color: white !important;
    }
    .traffic-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .traffic-low {
        background-color: #DCEDC8;
        color: #33691E;
    }
    .traffic-medium {
        background-color: #FFE0B2;
        color: #E65100;
    }
    .traffic-high {
        background-color: #FFCDD2;
        color: #B71C1C;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #E55A24;
    }
    .stSelectbox>div>div {
        background-color: white;
        border-radius: 4px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #9e9e9e;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stTable"] {
        border-radius: 10px;
    }
    .stDataFrame thead tr th {
        background-color: #FF6B35 !important;
        color: white !important;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #fff8f3;
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

def visualize_graph(G, path=None, title="NCR Traffic Network"):
    """Visualize the graph with optional path highlighting"""
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create a colormap for traffic levels
    edge_colors = []
    edge_widths = []
    
    for u, v in G.edges():
        traffic = G[u][v]['traffic']
        if traffic < 0.3:
            edge_colors.append('#4CAF50')  # Green for low traffic
        elif traffic < 0.7:
            edge_colors.append('#FF9800')  # Orange for medium traffic
        else:
            edge_colors.append('#F44336')  # Red for high traffic
        
        edge_widths.append(2 + traffic * 3)  # Width based on traffic
    
    # Draw nodes with a better style
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='#FF6B35', 
                          alpha=0.9, edgecolors='white', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                          arrowsize=15, connectionstyle='arc3,rad=0.1', alpha=0.7)
    
    # Highlight path if provided
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=5, 
                              edge_color='#3F51B5', arrowsize=20, alpha=1.0)
    
    # Draw labels with better styling
    node_labels = {node: f"{G.nodes[node]['name']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, 
                           font_weight='bold', font_color='white')
    
    # Draw edge labels with better formatting
    edge_labels = {}
    for u, v in G.edges():
        traffic_pct = int(G[u][v]['traffic'] * 100)
        if traffic_pct < 30:
            traffic_text = f"🟢 {traffic_pct}%"
        elif traffic_pct < 70:
            traffic_text = f"🟠 {traffic_pct}%"
        else:
            traffic_text = f"🔴 {traffic_pct}%"
        
        edge_labels[(u, v)] = f"{G[u][v]['name']}\n{G[u][v]['distance']}km, {traffic_text}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                font_color='#333333', bbox=dict(facecolor='white', edgecolor='none', 
                                                              alpha=0.7, pad=2))
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.axis('off')
    plt.tight_layout()
    return plt

def create_animated_graph(G, path=None):
    """Create an animated graph showing traffic flow"""
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_size=700, node_color='#FF6B35', 
                                  alpha=0.9, edgecolors='white', linewidths=2, ax=ax)
    
    # Draw labels
    node_labels = {node: f"{G.nodes[node]['name']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, 
                           font_weight='bold', font_color='white', ax=ax)
    
    # Initialize edges with traffic colors
    edges = {}
    for u, v in G.edges():
        traffic = G[u][v]['traffic']
        if traffic < 0.3:
            color = '#4CAF50'  # Green for low traffic
        elif traffic < 0.7:
            color = '#FF9800'  # Orange for medium traffic
        else:
            color = '#F44336'  # Red for high traffic
        
        width = 2 + traffic * 3
        edges[(u, v)] = nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                             width=width, edge_color=color, 
                                             arrowsize=15, connectionstyle='arc3,rad=0.1', 
                                             alpha=0.7, ax=ax)
    
    # Highlight path if provided
    path_edges = []
    if path:
        path_edges = list(zip(path, path[1:]))
        path_lines = nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=5, 
                                          edge_color='#3F51B5', arrowsize=20, 
                                          alpha=1.0, ax=ax)
    
    # Draw edge labels
    edge_labels = {}
    for u, v in G.edges():
        traffic_pct = int(G[u][v]['traffic'] * 100)
        if traffic_pct < 30:
            traffic_text = f"🟢 {traffic_pct}%"
        elif traffic_pct < 70:
            traffic_text = f"🟠 {traffic_pct}%"
        else:
            traffic_text = f"🔴 {traffic_pct}%"
        
        edge_labels[(u, v)] = f"{G[u][v]['name']}\n{G[u][v]['distance']}km, {traffic_text}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                font_color='#333333', bbox=dict(facecolor='white', 
                                                              edgecolor='none', 
                                                              alpha=0.7, pad=2), ax=ax)
    
    # Add vehicles on the path
    vehicles = []
    if path and len(path) > 1:
        # Create a vehicle (car icon) at the start of the path
        vehicle = ax.plot([], [], 'o', color='#3F51B5', markersize=10, 
                         markeredgecolor='white', markeredgewidth=1)[0]
        vehicles.append((vehicle, path))
    
    ax.set_title("NCR Traffic Network - Live Simulation", fontsize=16, 
                fontweight='bold', pad=20, color='#333333')
    ax.axis('off')
    plt.tight_layout()
    
    # Animation function
    def update(frame):
        # Update traffic levels
        for u, v in G.edges():
            # Simulate traffic fluctuation
            traffic_change = np.sin(frame/10 + hash((u, v)) % 10) * 0.05
            new_traffic = max(0.1, min(0.9, G[u][v]['traffic'] + traffic_change))
            G[u][v]['traffic'] = new_traffic
            
            # Update edge color based on new traffic
            if new_traffic < 0.3:
                color = '#4CAF50'  # Green
            elif new_traffic < 0.7:
                color = '#FF9800'  # Orange
            else:
                color = '#F44336'  # Red
            
            width = 2 + new_traffic * 3
            
            # Skip updating path edges
            if path and (u, v) in path_edges:
                continue
                
            # Update edge appearance
            edges[(u, v)][0].set_color(color)
            edges[(u, v)][0].set_linewidth(width)
        
        # Update vehicle positions
        for vehicle, vehicle_path in vehicles:
            if len(vehicle_path) < 2:
                continue
                
            # Calculate position along the path
            path_position = frame % (len(vehicle_path) - 1)
            path_index = int(path_position)
            path_fraction = path_position - path_index
            
            # Get the current edge
            u = vehicle_path[path_index]
            v = vehicle_path[path_index + 1]
            
            # Interpolate position
            start_x, start_y = pos[u]
            end_x, end_y = pos[v]
            
            x = start_x + path_fraction * (end_x - start_x)
            y = start_y + path_fraction * (end_y - start_y)
            
            vehicle.set_data([x], [y])
        
        return [edges[(u, v)][0] for u, v in G.edges()] + [vehicle for vehicle, _ in vehicles]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)
    
    # Convert animation to HTML5 video
    plt.close(fig)  # Prevent display of the figure
    
    return ani

def create_plotly_animated_graph(G, path=None):
    """Create an interactive animated graph using Plotly"""
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_widths = []
    edge_texts = []
    
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Add edge coordinates
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Determine color based on traffic
        traffic = G[u][v]['traffic']
        if traffic < 0.3:
            color = 'rgba(76, 175, 80, 0.7)'  # Green
        elif traffic < 0.7:
            color = 'rgba(255, 152, 0, 0.7)'  # Orange
        else:
            color = 'rgba(244, 67, 54, 0.7)'  # Red
        
        # Determine if this edge is in the path
        is_path_edge = path and u in path and v in path and path.index(u) == path.index(v) - 1
        
        if is_path_edge:
            color = 'rgba(63, 81, 181, 1.0)'  # Blue for path
            width = 5
        else:
            width = 2 + traffic * 3
        
        # Add color and width for each segment (including None)
        edge_colors.extend([color, color, color])
        edge_widths.extend([width, width, width])
        
        # Add hover text
        traffic_pct = int(traffic * 100)
        edge_text = f"Road: {G[u][v]['name']}<br>Distance: {G[u][v]['distance']} km<br>Traffic: {traffic_pct}%"
        edge_texts.extend([edge_text, edge_text, ""])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        line_color=edge_colors,
        line_width=edge_widths,
        text=edge_texts
    )
    
    # Create nodes
    node_x = []
    node_y = []
    node_texts = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_texts.append(f"City: {G.nodes[node]['name']}")
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[node]['name'] for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color='black'),
        marker=dict(
            showscale=False,
            color='#FF6B35',
            size=20,
            line_width=2,
            line=dict(color='white')
        ),
        hovertext=node_texts
    )
    
    # Create vehicle trace if path is provided
    vehicle_trace = None
    if path and len(path) > 1:
        # Start at the first node in the path
        start_node = path[0]
        x, y = pos[start_node]
        
        vehicle_trace = go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(
                color='#3F51B5',
                size=15,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            hoverinfo='text',
            hovertext='Vehicle',
            name='Vehicle'
        )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace] + ([vehicle_trace] if vehicle_trace else []),
        layout=go.Layout(
            title='NCR Traffic Network - Interactive Simulation',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(248,249,250,1)',
            paper_bgcolor='rgba(248,249,250,1)',
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                )]
            )]
        )
    )
    
    # Create animation frames if path is provided
    if path and len(path) > 1:
        frames = []
        
        for i in range(50):  # 50 frames of animation
            # Calculate position along the path
            path_length = len(path) - 1
            path_position = (i % path_length)
            path_index = int(path_position)
            path_fraction = path_position - path_index
            
            # Get the current edge
            u = path[path_index]
            v = path[path_index + 1]
            
            # Interpolate position
            start_x, start_y = pos[u]
            end_x, end_y = pos[v]
            
            x = start_x + path_fraction * (end_x - start_x)
            y = start_y + path_fraction * (end_y - start_y)
            
            # Create a frame with updated vehicle position
            frame = go.Frame(
                data=[edge_trace, node_trace, 
                     go.Scatter(x=[x], y=[y], mode='markers', 
                               marker=dict(color='#3F51B5', size=15, symbol='circle',
                                          line=dict(color='white', width=2)))],
                name=f"frame{i}"
            )
            frames.append(frame)
        
        fig.frames = frames
    
    return fig

def create_folium_map(G, path=None):
    """Create an interactive folium map"""
    # Create a base map centered on Delhi (approximate coordinates)
    base_lat, base_lon = 28.6139, 77.2090  # Delhi coordinates
    scale = 0.05  # Scale factor for visualization
    
    m = folium.Map(location=[base_lat, base_lon], zoom_start=9, tiles="CartoDB positron")
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Add nodes as markers
    for node, position in pos.items():
        lat = base_lat + position[0] * scale
        lon = base_lon + position[1] * scale
        
        popup_text = f"<b>{G.nodes[node]['name']}</b>"
        
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
            weight = 3
            opacity = 0.7
        elif traffic < 0.7:
            color = 'orange'
            weight = 4
            opacity = 0.8
        else:
            color = 'red'
            weight = 5
            opacity = 0.9
        
        # Highlight path edges
        if path and u in path and v in path and path.index(u) == path.index(v) - 1:
            color = 'purple'
            weight = 6
            opacity = 1.0
        
        popup_text = f"""
        <div style='font-family: Arial; font-size: 12px;'>
            <b>{data['name']}</b><br>
            Distance: {data['distance']} km<br>
            Traffic: {int(traffic*100)}%
        </div>
        """
        
        folium.PolyLine(
            locations=[[u_lat, u_lon], [v_lat, v_lon]],
            popup=folium.Popup(popup_text, max_width=200),
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=f"{data['name']} - {int(traffic*100)}% traffic"
        ).add_to(m)
    
    # Add animated vehicle if path is provided
    if path and len(path) > 1:
        # Create a feature group for the vehicle
        vehicle_group = folium.FeatureGroup(name="Vehicle")
        
        # Add the vehicle marker at the start position
        start_node = path[0]
        start_pos = pos[start_node]
        start_lat = base_lat + start_pos[0] * scale
        start_lon = base_lon + start_pos[1] * scale
        
        vehicle_marker = folium.Marker(
            location=[start_lat, start_lon],
            icon=folium.Icon(color='blue', icon='car', prefix='fa'),
            tooltip="Vehicle"
        )
        vehicle_group.add_child(vehicle_marker)
        m.add_child(vehicle_group)
    
    return m

def simulate_traffic_change():
    """Simulate traffic changes over time"""
    data = load_sample_data()
    
    # Get current time to simulate rush hour effects
    current_hour = datetime.now().hour
    
    # Rush hour factors (morning and evening rush hours)
    is_morning_rush = 8 <= current_hour <= 10
    is_evening_rush = 17 <= current_hour <= 19
    
    for road in data["roads"]:
        # Base random adjustment
        change = random.uniform(-0.15, 0.15)
        
        # Apply rush hour effects
        if is_morning_rush:
            # Increase traffic into major cities during morning rush
            if road["to"] == "A" or road["to"] == "B" or road["to"] == "C":  # Delhi, Gurgaon, Noida
                change += 0.2
        elif is_evening_rush:
            # Increase traffic out of major cities during evening rush
            if road["from"] == "A" or road["from"] == "B" or road["from"] == "C":
                change += 0.2
        
        # Apply the change with limits
        road["traffic"] = max(0.1, min(0.95, road["traffic"] + change))
    
    return data

def get_traffic_badge(traffic_level):
    """Return HTML for a traffic badge based on level"""
    level = int(traffic_level * 100)
    if level < 30:
        return f'<span class="traffic-badge traffic-low">{level}%</span>'
    elif level < 70:
        return f'<span class="traffic-badge traffic-medium">{level}%</span>'
    else:
        return f'<span class="traffic-badge traffic-high">{level}%</span>'

def main():
    # Sidebar
    st.sidebar.markdown('<p class="main-header">🚦 Smart Traffic</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Flow Optimizer</p>', unsafe_allow_html=True)
    
    # Add time of day indicator in sidebar
    current_time = datetime.now().strftime("%H:%M")
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        time_greeting = "Good Morning"
        time_icon = "🌅"
    elif 12 <= current_hour < 17:
        time_greeting = "Good Afternoon"
        time_icon = "☀️"
    elif 17 <= current_hour < 21:
        time_greeting = "Good Evening"
        time_icon = "🌆"
    else:
        time_greeting = "Good Night"
        time_icon = "🌙"
    
    st.sidebar.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <div style="font-size: 0.8rem; color: #666;">Current Time</div>
        <div style="font-size: 1.2rem; font-weight: bold;">{time_icon} {current_time}</div>
        <div style="font-size: 0.9rem;">{time_greeting}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Navigation")
    
    # App sections
    tab1, tab2, tab3, tab4 = st.tabs(["🚗 Route Optimizer", "🔄 Traffic Simulation", "📊 Algorithm Comparison", "ℹ️ About"])
    
    with tab1:
        st.markdown('<p class="main-header">🚗 Smart Route Optimizer</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Find the optimal route between two locations in the National Capital Region (NCR) considering current traffic conditions.
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
                        
                        # Estimate travel time (in minutes)
                        # Assuming average speed of 60 km/h with no traffic, reduced by traffic factor
                        avg_speed = 60 * (1 - avg_traffic * 0.7)  # km/h
                        travel_time = (total_distance / avg_speed) * 60  # minutes
                        
                        # Display metrics
                        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### Route Summary")
                        
                        # Create a more visual path representation
                        path_cities = [f"{G.nodes[node]['name']}" for node in path]
                        path_display = " → ".join(path_cities)
                        st.markdown(f"**Route:** {path_display}")
                        
                        col1a, col2a, col3a = st.columns(3)
                        with col1a:
                            st.metric("Total Distance", f"{total_distance:.1f} km")
                        with col2a:
                            traffic_html = get_traffic_badge(avg_traffic)
                            st.markdown(f"**Traffic Level:**<br>{traffic_html}", unsafe_allow_html=True)
                        with col3a:
                            st.metric("Est. Travel Time", f"{travel_time:.0f} min")
                        
                        st.markdown(f"**Computation Time:** {computation_time*1000:.2f} ms")
                        
                        # Display route directions
                        st.markdown("### Turn-by-Turn Directions")
                        for i, road in enumerate(road_names):
                            from_city = G.nodes[path[i]]['name']
                            to_city = G.nodes[path[i+1]]['name']
                            traffic_level = G[path[i]][path[i+1]]['traffic']
                            traffic_html = get_traffic_badge(traffic_level)
                            
                            st.markdown(f"{i+1}. Take <span class='highlight'>{road}</span> from {from_city} to {to_city} ({G[path[i]][path[i+1]]['distance']} km) {traffic_html}", unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"No path found between {source_node} and {dest_node}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### NCR Traffic Network")
            
            # Create tabs for different visualizations
            map_tab1, map_tab2, map_tab3 = st.tabs(["Network Graph", "Interactive Map", "Animated Route"])
            
            with map_tab1:
                # Visualize the graph
                fig = visualize_graph(G, path=path if 'path' in locals() else None)
                st.pyplot(fig)
            
            with map_tab2:
                # Create interactive map
                m = create_folium_map(G, path=path if 'path' in locals() else None)
                folium_static(m)
            
            with map_tab3:
                if 'path' in locals() and path:
                    st.markdown("### Route Animation")
                    # Create animated graph using Plotly
                    fig = create_plotly_animated_graph(G, path)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Calculate a route first to see the animation")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<p class="main-header">🔄 Traffic Simulation</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Simulate changing traffic conditions in the NCR and observe how they affect optimal routes.
        This helps understand the dynamic nature of urban traffic patterns throughout the day.
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
            
            # Create a more visual traffic table
            roads_df = pd.DataFrame(data_to_display["roads"])
            
            # Add city names instead of just node IDs
            roads_df['from_city'] = roads_df['from'].apply(lambda x: data_to_display["intersections"][x]["name"])
            roads_df['to_city'] = roads_df['to'].apply(lambda x: data_to_display["intersections"][x]["name"])
            
            # Format traffic as HTML badges
            roads_df['traffic_html'] = roads_df['traffic'].apply(get_traffic_badge)
            
            # Create a styled dataframe
            styled_df = pd.DataFrame({
                'Road': roads_df['name'],
                'From': roads_df['from_city'],
                'To': roads_df['to_city'],
                'Distance (km)': roads_df['distance'],
                'Traffic': roads_df['traffic_html']
            })
            
            # Display with HTML rendering for the badges
            st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Traffic Visualization")
            
            # Create graph from current data
            current_data = st.session_state.get('simulation_data', load_sample_data())
            G = create_graph_from_data(current_data, consider_traffic=True)
            
            # Create tabs for static and animated visualizations
            vis_tab1, vis_tab2 = st.tabs(["Static View", "Animated Simulation"])
            
            with vis_tab1:
                # Visualize
                fig = visualize_graph(G, title="Current NCR Traffic Conditions")
                st.pyplot(fig)
            
            with vis_tab2:
                # Create animated visualization
                fig = create_plotly_animated_graph(G)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<p class="main-header">📊 Algorithm Comparison</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Compare different routing algorithms to understand their performance characteristics.
        This helps in selecting the most appropriate algorithm for specific traffic scenarios in the NCR.
        </p>
        """, unsafe_allow_html=True)
        
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
                
                # Format paths with city names
                def format_path(path):
                    if not path:
                        return "No path found"
                    return " → ".join([f"{data['intersections'][node]['name']}" for node in path])
                
                # Collect results
                results.append({
                    "Algorithm": "Dijkstra's Algorithm",
                    "Path": format_path(dijkstra_path),
                    "Distance (km)": f"{dijkstra_dist:.2f}" if dijkstra_path else "N/A",
                    "Computation Time (ms)": f"{dijkstra_time*1000:.2f}",
                    "Path Length": len(dijkstra_path) - 1 if dijkstra_path else 0
                })
                
                results.append({
                    "Algorithm": "A* Algorithm",
                    "Path": format_path(astar_path),
                    "Distance (km)": f"{astar_dist:.2f}" if astar_path else "N/A",
                    "Computation Time (ms)": f"{astar_time*1000:.2f}",
                    "Path Length": len(astar_path) - 1 if astar_path else 0
                })
                
                results.append({
                    "Algorithm": "Bellman-Ford Algorithm",
                    "Path": format_path(bellman_ford_path),
                    "Distance (km)": f"{bellman_ford_dist:.2f}" if bellman_ford_path else "N/A",
                    "Computation Time (ms)": f"{bellman_ford_time*1000:.2f}",
                    "Path Length": len(bellman_ford_path) - 1 if bellman_ford_path else 0
                })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Create bar chart for computation time
                fig = px.bar(
                    results_df, 
                    x="Algorithm", 
                    y=[float(t.split()[0]) for t in results_df["Computation Time (ms)"]],
                    labels={"y": "Computation Time (ms)"},
                    title="Algorithm Performance Comparison",
                    color="Algorithm",
                    color_discrete_sequence=["#FF6B35", "#4CAF50", "#3F51B5"]
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='rgba(248,249,250,1)',
                    font=dict(size=12),
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
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
                
                # Visualize all paths on the same graph for comparison
                if dijkstra_path and astar_path and bellman_ford_path:
                    st.markdown("### Visual Path Comparison")
                    
                    # Create a copy of the graph for visualization
                    G_viz = G.copy()
                    
                    # Create a figure with subplots
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Get positions
                    pos = nx.get_node_attributes(G, 'pos')
                    
                    # Plot each algorithm's path
                    for i, (path, title) in enumerate([
                        (dijkstra_path, "Dijkstra's Algorithm"),
                        (astar_path, "A* Algorithm"),
                        (bellman_ford_path, "Bellman-Ford Algorithm")
                    ]):
                        # Draw nodes
                        nx.draw_networkx_nodes(G_viz, pos, node_size=500, 
                                              node_color='#FF6B35', alpha=0.8, 
                                              ax=axes[i])
                        
                        # Draw all edges
                        nx.draw_networkx_edges(G_viz, pos, width=1, alpha=0.3, 
                                              edge_color='gray', ax=axes[i])
                        
                        # Highlight path
                        path_edges = list(zip(path, path[1:]))
                        nx.draw_networkx_edges(G_viz, pos, edgelist=path_edges, 
                                              width=3, edge_color='#3F51B5', 
                                              ax=axes[i])
                        
                        # Draw labels
                        nx.draw_networkx_labels(G_viz, pos, font_size=8, 
                                               font_weight='bold', ax=axes[i])
                        
                        axes[i].set_title(title)
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### Smart Traffic Flow Optimization System for NCR
        
        This application demonstrates the use of graph algorithms for optimizing traffic flow in the National Capital Region (NCR) of India.
        It implements several key algorithms covered in the course:
        
        - **Dijkstra's Algorithm**: A greedy algorithm that finds the shortest path between nodes in a graph
        - **A* Algorithm**: An extension of Dijkstra's that uses heuristics to speed up the search
        - **Bellman-Ford Algorithm**: An algorithm that computes shortest paths from a single source vertex to all other vertices
        
        ### Course Outcomes Addressed
        
        1. **CO1**: The application demonstrates asymptotic notations by analyzing algorithm complexity
        2. **CO2**: It implements various algorithm paradigms including greedy algorithms
        3. **CO5**: It applies Dijkstra's, Bellman-Ford, and other algorithms to solve real-world problems like traffic routing in the NCR
        
        ### Technologies Used
        
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **NetworkX**: Graph manipulation and analysis
        - **Matplotlib & Plotly**: Data visualization and animations
        - **Folium**: Interactive maps
        - **Pandas**: Data manipulation
        
        ### Future Enhancements
        
        - Integration with real-time traffic data APIs from Indian traffic authorities
        - Machine learning models to predict traffic patterns in the NCR
        - More sophisticated traffic simulation models based on historical Delhi traffic data
        - Support for multi-modal transportation routing including Delhi Metro
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Smart Traffic Flow Optimization System | Developed with ❤️ using Python & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
