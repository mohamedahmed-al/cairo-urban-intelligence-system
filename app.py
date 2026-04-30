import streamlit as st
import pandas as pd
import networkx as nx
import math
import time
from pyvis.network import Network
import streamlit.components.v1 as components

# 1. Page Configuration
st.set_page_config(page_title="Cairo Urban Intelligence", layout="wide")

# 2. Data Loading & Graph Construction
@st.cache_data
def load_and_build_graph():
    G = nx.Graph()
    id_to_name = {}
    
    try:
        # Loading data from csvFiles folder
        df_n = pd.read_csv("csvFiles/Geographic_Data(Neighborhoods_and_Districts).csv")
        df_f = pd.read_csv("csvFiles/Geographic_Data(Important_Facilities).csv")
        df_r = pd.read_csv("csvFiles/Road_Network_Data(Existing_Roads).csv")

        # Adding Neighborhood Nodes
        for _, row in df_n.iterrows():
            nid = str(row['ID']).strip()
            id_to_name[nid] = row['Name']
            G.add_node(nid, name=row['Name'], label=row['Name'], 
                       group="Neighborhood", x=row['X-coordinate'], y=row['Y-coordinate'])

        # Adding Facility Nodes
        for _, row in df_f.iterrows():
            fid = str(row['ID']).strip()
            id_to_name[fid] = row['Name']
            G.add_node(fid, name=row['Name'], label=row['Name'], 
                       group="Facility", x=row['X-coordinate'], y=row['Y-coordinate'])

        # Adding Edges (Roads) - Greedy Implementation Base
        for _, row in df_r.iterrows():
            G.add_edge(str(row['FromID']).strip(), str(row['TOID']).strip(), weight=row['Distance(km)'])
            
    except Exception as e:
        st.error(f"⚠️ Data Loading Error: {e}")
        
    return G, id_to_name

cairo_network, id_to_name = load_and_build_graph()

# 3. Interactive Map Generation Function
def generate_map_html(graph, path=None):
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#000000")
    for node, data in graph.nodes(data=True):
        is_in_path = path and node in path
        color = "#e74c3c" if is_in_path else ("#3498db" if data.get('group') == "Neighborhood" else "#2ecc71")
        net.add_node(node, label=data.get('name', node), color=color, size=25 if is_in_path else 15)

    for u, v, data in graph.edges(data=True):
        is_path_edge = False
        if path:
            path_edges = list(zip(path, path[1:]))
            if (u, v) in path_edges or (v, u) in path_edges:
                is_path_edge = True
        net.add_edge(u, v, width=5 if is_path_edge else 1, color="#e74c3c" if is_path_edge else "#bdc3c7")

    net.set_options('{"physics": {"enabled": true, "solver": "forceAtlas2Based"}}')
    return net.generate_html()

# 4. Main User Interface
st.title("🏙️ Cairo Urban Intelligence System")
st.markdown("---")

# Tabs for Algorithm Categories
tab1, tab2, tab3 = st.tabs(["🚀 Shortest Path", "🏗️ Infrastructure (Greedy)", "📊 Optimization (DP)"])

# Tab 1: Shortest Path (Dijkstra vs A*)
with tab1:
    st.header("Pathfinding Comparison")
    col_a, col_b = st.columns(2)
    options = {v: k for k, v in id_to_name.items()}
    
    if options:
        start_point = col_a.selectbox("Select Origin", list(options.keys()), index=0)
        end_point = col_b.selectbox("Select Destination", list(options.keys()), index=min(4, len(options)-1))
        s_id, e_id = options[start_point], options[end_point]

        if st.button("🏁 Run Algorithms"):
            c1, c2 = st.columns(2)
            
            # Dijkstra
            with c1:
                st.subheader("Dijkstra's Algorithm")
                try:
                    t0 = time.perf_counter()
                    d_path = nx.shortest_path(cairo_network, s_id, e_id, weight='weight')
                    st.success(f"⏱️ Time: {(time.perf_counter()-t0)*1000:.2f}ms | 📏 Dist: {nx.path_weight(cairo_network, d_path, 'weight'):.2f}km")
                    components.html(generate_map_html(cairo_network, d_path), height=500)
                except nx.NetworkXNoPath:
                    st.error("No path exists between these points.")

            # A* Search
            with c2:
                st.subheader("A* Search (Heuristic)")
                try:
                    def haversine_dist(u, v):
                        n1, n2 = cairo_network.nodes[u], cairo_network.nodes[v]
                        return math.sqrt((n1['x']-n2['x'])**2 + (n1['y']-n2['y'])**2) * 111
                    t0 = time.perf_counter()
                    a_path = nx.astar_path(cairo_network, s_id, e_id, heuristic=haversine_dist, weight='weight')
                    st.success(f"⏱️ Time: {(time.perf_counter()-t0)*1000:.2f}ms | 📏 Dist: {nx.path_weight(cairo_network, a_path, 'weight'):.2f}km")
                    components.html(generate_map_html(cairo_network, a_path), height=500)
                except nx.NetworkXNoPath:
                    st.error("No path exists between these points.")
    else:
        st.warning("Data not found. Please check 'csvFiles' directory.")

# Tab 2: Minimum Spanning Tree (Greedy)
with tab2:
    st.header("Optimal Network Design (MST)")
    st.info("Utilizing Kruskal's Greedy Algorithm to find the minimum distance required to connect all points.")
    if st.button("🏗️ Build MST"):
        try:
            mst = nx.minimum_spanning_tree(cairo_network.to_undirected(), weight='weight')
            st.metric("Total Spanning Distance", f"{mst.size(weight='weight'):.2f} km")
            components.html(generate_map_html(mst), height=600)
        except Exception as e:
            st.error(f"Calculation Error: {e}")

# Tab 3: Dynamic Programming (Resource Allocation)
with tab3:
    st.header("Urban Budget Optimization")
    st.write("Solving the 0/1 Knapsack Problem using Dynamic Programming to maximize maintenance utility.")
    
    budget = st.slider("Maintenance Budget (Million EGP)", 10, 200, 75)
    
    # Static Data for DP Problem
    costs = [20, 30, 50, 40, 15, 45, 25, 55, 60, 20] 
    utility = [45, 80, 110, 90, 35, 100, 60, 130, 140, 50]

    def knapsack_dp(W, wt, val):
        n = len(val)
        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for w in range(W + 1):
                if wt[i-1] <= w:
                    dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        return dp[n][W]

    if st.button("📊 Calculate Optimal ROI"):
        max_score = knapsack_dp(budget, costs, utility)
        st.success(f"Maximum Calculated Urban Utility Score: {max_score}")
        st.info("This logic ensures optimal resource distribution across Cairo's infrastructure projects.")
