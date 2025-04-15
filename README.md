# Smart Traffic Flow Optimization System

A web application that leverages graph algorithms to model and optimize urban traffic networks. This system aims to improve vehicle routing, minimize congestion, and reduce average commute times using simulated traffic data.

## Features

- **Route Optimization**: Find the optimal route between two locations considering current traffic conditions
- **Traffic Simulation**: Simulate changing traffic conditions and observe their effects on optimal routes
- **Algorithm Comparison**: Compare different routing algorithms (Dijkstra's, A*, Bellman-Ford) to understand their performance characteristics
- **Interactive Visualization**: View the traffic network as both a graph and an interactive map

## Course Outcomes Addressed

1. Application of graph-based algorithms for real-world problem-solving
2. Development of efficient path-finding logic using adjacency structures
3. Translation of algorithmic concepts into visually interactive applications
4. Integration of frontend and backend logic in a full-stack project
5. Deployment of Python-based apps on cloud platforms

## Installation and Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/Vidish-Bijalwan/SMART-TRAFFIC-OPTIMIZER.git
cd smart-traffic-optimizer
\`\`\`

2. Install the required dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Run the Streamlit app:
\`\`\`bash
streamlit run app.py
\`\`\`

## Project Structure

\`\`\`
smart-traffic-optimizer/
│
├── .streamlit/                   # Streamlit configuration
│   └── config.toml
├── algorithms/                   # Graph algorithms
│   ├── dijkstra.py
│   ├── astar.py
│   ├── bellman_ford.py
│   └── utils.py
├── data/                         # Sample city map/road data
│   └── city_graph.json
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
\`\`\`

## Algorithms Implemented

- **Dijkstra's Algorithm**: A greedy algorithm that finds the shortest path between nodes in a graph
- **A* Algorithm**: An extension of Dijkstra's that uses heuristics to speed up the search
- **Bellman-Ford Algorithm**: An algorithm that computes shortest paths from a single source vertex to all other vertices


## Future Enhancements

- Integration with real-time traffic data APIs
- Machine learning models to predict traffic patterns
- More sophisticated traffic simulation models
- Support for multi-modal transportation routing

## License

MIT
# SMART-TRAFFIC-OPTIMIZER

**Deployed site Link** : https://smart-traffic-optimizer-vidish-bijalwan.streamlit.app/
