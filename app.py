import streamlit as st
import pandas as pd
import networkx as nx
import math
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
 
# 1. Page Configuration
st.set_page_config(page_title="Cairo Urban Intelligence", layout="wide", page_icon="🏙️")
 
# 2. Data Loading & Graph Construction
@st.cache_data
def load_and_build_graph():
    G = nx.Graph()
    id_to_name = {}
    try:
        df_n = pd.read_csv("csvFiles/Geographic_Data(Neighborhoods_and_Districts).csv")
        df_f = pd.read_csv("csvFiles/Geographic_Data(Important_Facilities).csv")
        df_r = pd.read_csv("csvFiles/Road_Network_Data(Existing_Roads).csv")
 
        for _, row in df_n.iterrows():
            nid = str(row['ID']).strip()
            id_to_name[nid] = row['Name']
            G.add_node(nid, name=row['Name'], label=row['Name'],
                       group="Neighborhood", x=row['X-coordinate'], y=row['Y-coordinate'])
 
        for _, row in df_f.iterrows():
            fid = str(row['ID']).strip()
            id_to_name[fid] = row['Name']
            G.add_node(fid, name=row['Name'], label=row['Name'],
                       group="Facility", x=row['X-coordinate'], y=row['Y-coordinate'])
 
        for _, row in df_r.iterrows():
            G.add_edge(str(row['FromID']).strip(), str(row['TOID']).strip(), weight=row['Distance(km)'])
 
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
    return G, id_to_name
 
 
@st.cache_data
def load_traffic_data():
    return pd.read_csv("csvFiles/Traffic_Flow_Data_Patterns.csv")
 
 
cairo_network, id_to_name = load_and_build_graph()
traffic_df = load_traffic_data()
 
 
# 3. Map Generation
def generate_map_html(graph, path=None):
    net = Network(height="480px", width="100%", bgcolor="#ffffff", font_color="#000000")
    for node, data in graph.nodes(data=True):
        in_path = path and node in path
        if in_path:
            color = "#e74c3c"
        elif data.get('group') == "Neighborhood":
            color = "#3498db"
        else:
            color = "#2ecc71"
        net.add_node(node, label=data.get('name', node), color=color,
                     size=28 if in_path else 14)
 
    for u, v, data in graph.edges(data=True):
        is_path_edge = False
        if path:
            path_edges = list(zip(path, path[1:]))
            if (u, v) in path_edges or (v, u) in path_edges:
                is_path_edge = True
        net.add_edge(u, v, width=4 if is_path_edge else 1,
                     color="#e74c3c" if is_path_edge else "#bdc3c7")
 
    net.set_options('{"physics":{"enabled":true,"solver":"forceAtlas2Based"}}')
    return net.generate_html()
 
 
# 4. ML Traffic Prediction
@st.cache_resource
def train_ml_models(df):
    time_cols  = ['Morning Peak(veh/h)', 'Afternoon (veh/h)', 'Evening Peak(veh/h)', 'Night(veh/h)']
    time_hours = [8, 13, 18, 23]
    time_idx_map = {c: i for i, c in enumerate(time_cols)}
 
    rows = []
    for i, row in df.iterrows():
        parts = str(row['RoadID']).replace('F', '9').split('_')
        try:
            n1, n2 = int(parts[0]), int(parts[1])
        except:
            n1, n2 = 5, 5
        road_avg = row[time_cols].mean()
        for col in time_cols:
            t_idx = time_idx_map[col]
            rows.append({
                'hour': time_hours[t_idx],
                'time_idx': t_idx,
                'node1': n1,
                'node2': n2,
                'road_avg': road_avg,
                'volume': row[col]
            })
 
    data = pd.DataFrame(rows)
    features = ['hour', 'time_idx', 'node1', 'node2', 'road_avg']
    X, y = data[features], data['volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
 
    models = {
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
    }
 
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        results[name] = {
            'model': model, 'scaler': scaler,
            'mae': mean_absolute_error(y_test, preds),
            'r2':  r2_score(y_test, preds),
            'preds': preds, 'y_test': y_test.values,
        }
 
    return results, features, time_hours
 
 
ml_results, ml_features, time_hours = train_ml_models(traffic_df)
 
 
# 5. Helpers
def haversine_dist(u, v):
    n1, n2 = cairo_network.nodes[u], cairo_network.nodes[v]
    return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2) * 111
 
 
def knapsack_dp(W, wt, val):
    n = len(val)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if wt[i-1] <= w:
                dp[i][w] = max(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    selected, w = [], W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i - 1)
            w -= wt[i-1]
    return dp[n][W], selected
 
 
# 6. Header
st.title("🏙️ Cairo Urban Intelligence System")
st.markdown("---")
 
# 7. Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚔️ Algorithm Race",
    "🏗️ Infrastructure (MST)",
    "📊 Budget Optimization (DP)",
    "🤖 ML Traffic Prediction",
    "⏱️ Time-Varying Routes",
    "🚦 Signal Optimization",
    "🚑 Transit Scheduling (DP)",
])
 
 
# TAB 1 — SIDE-BY-SIDE ALGORITHM RACE
with tab1:
    st.header("Dijkstra vs A* — Side-by-Side Comparison")
 
    options = {v: k for k, v in id_to_name.items()}
 
    if options:
        col_a, col_b = st.columns(2)
        start_point = col_a.selectbox("Select Origin", list(options.keys()), index=0)
        end_point   = col_b.selectbox("Select Destination", list(options.keys()),
                                      index=min(4, len(options) - 1))
        s_id, e_id  = options[start_point], options[end_point]
 
        if st.button("🏁 Run Race"):
            d_path = d_time = d_dist = None
            a_path = a_time = a_dist = None
 
            try:
                t0 = time.perf_counter()
                d_path = nx.shortest_path(cairo_network, s_id, e_id, weight='weight')
                d_time = (time.perf_counter() - t0) * 1000
                d_dist = nx.path_weight(cairo_network, d_path, 'weight')
            except nx.NetworkXNoPath:
                pass
 
            try:
                t0 = time.perf_counter()
                a_path = nx.astar_path(cairo_network, s_id, e_id,
                                       heuristic=haversine_dist, weight='weight')
                a_time = (time.perf_counter() - t0) * 1000
                a_dist = nx.path_weight(cairo_network, a_path, 'weight')
            except nx.NetworkXNoPath:
                pass
 
            # Scoreboard
            if d_time and a_time:
                winner  = "A* Search" if a_time < d_time else "Dijkstra"
                speedup = max(d_time, a_time) / min(d_time, a_time)
                st.success(f"🏆 Winner: **{winner}** — {speedup:.1f}x faster")
 
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Dijkstra Time", f"{d_time:.3f} ms")
                sc2.metric("A* Time",       f"{a_time:.3f} ms")
                sc3.metric("Dijkstra Dist", f"{d_dist:.2f} km")
                sc4.metric("A* Dist",       f"{a_dist:.2f} km")
 
                fig_bar = go.Figure(go.Bar(
                    x=['Dijkstra', 'A* Search'],
                    y=[d_time, a_time],
                    marker_color=['#3498db', '#2ecc71'],
                    text=[f'{d_time:.3f}ms', f'{a_time:.3f}ms'],
                    textposition='outside',
                ))
                fig_bar.update_layout(
                    title=f"Speed Comparison — {winner} wins by {speedup:.1f}x",
                    yaxis_title='Time (ms)',
                    height=300
                )
                st.plotly_chart(fig_bar, width="stretch")
 
            # Side-by-side maps
            st.subheader("Route Visualization")
            map1, map2 = st.columns(2)
 
            with map1:
                st.markdown("**Dijkstra's Algorithm** — Explores all nodes uniformly")
                if d_path:
                    st.success(f"Nodes: {len(d_path)} | Dist: {d_dist:.2f} km | Time: {d_time:.3f} ms")
                    components.html(generate_map_html(cairo_network, d_path), height=490)
                else:
                    st.error("No path found")
 
            with map2:
                st.markdown("**A* Search** — Uses heuristic to guide search")
                if a_path:
                    st.success(f"Nodes: {len(a_path)} | Dist: {a_dist:.2f} km | Time: {a_time:.3f} ms")
                    components.html(generate_map_html(cairo_network, a_path), height=490)
                else:
                    st.error("No path found")
 
            with st.expander("📖 How do these algorithms differ?"):
                st.markdown("""
**Dijkstra's Algorithm** expands every reachable node in cost order — guarantees shortest path but may visit many unnecessary nodes.
 
**A* Search** adds a straight-line distance heuristic to prioritize nodes closer to the goal — typically faster while still finding the optimal path.
                """)
    else:
        st.warning("Data not found. Please check the 'csvFiles' directory.")
 
 
# TAB 2 — INFRASTRUCTURE / MST
with tab2:
    st.header("Optimal Network Design (MST)")
    st.info("Kruskal's Greedy Algorithm finds the minimum total road length to connect all network nodes.")
 
    if st.button("🏗️ Build MST"):
        try:
            mst = nx.minimum_spanning_tree(cairo_network.to_undirected(), weight='weight')
            total_dist = mst.size(weight='weight')
            original   = cairo_network.size(weight='weight')
 
            col1, col2, col3 = st.columns(3)
            col1.metric("MST Total Distance", f"{total_dist:.1f} km")
            col2.metric("Original Network",   f"{original:.1f} km")
            col3.metric("Redundancy Removed", f"{((original - total_dist) / original * 100):.1f}%")
 
            components.html(generate_map_html(mst), height=600)
        except Exception as e:
            st.error(f"Calculation Error: {e}")
 
 
# TAB 3 — DYNAMIC PROGRAMMING
with tab3:
    st.header("Urban Budget Optimization — 0/1 Knapsack (DP)")
    st.write("Maximize maintenance utility across Cairo's infrastructure using Dynamic Programming.")
 
    budget = st.slider("Maintenance Budget (Million EGP)", 10, 200, 75)
 
    project_names = [
        "Ring Road Resurfacing", "Metro Line Extension", "Nile Bridge Upgrade",
        "Smart Traffic Lights", "Bike Lane Network", "Flood Drainage System",
        "Street Lighting LED", "Pedestrian Overpasses", "Bus Rapid Transit",
        "Green Space Development"
    ]
    costs   = [20, 30, 50, 40, 15, 45, 25, 55, 60, 20]
    utility = [45, 80, 110, 90, 35, 100, 60, 130, 140, 50]
 
    with st.expander("View All Projects"):
        st.dataframe(pd.DataFrame({
            'Project': project_names,
            'Cost (M EGP)': costs,
            'Utility Score': utility,
            'ROI': [round(u / c, 2) for u, c in zip(utility, costs)]
        }), width="stretch")
 
    if st.button("📊 Calculate Optimal Allocation"):
        max_score, selected_idx = knapsack_dp(budget, costs, utility)
        total_cost = sum(costs[i] for i in selected_idx)
 
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Utility Score",  str(max_score))
        col2.metric("Budget Used",        f"{total_cost}M EGP")
        col3.metric("Projects Selected",  str(len(selected_idx)))
 
        st.subheader("Selected Projects")
        for i in selected_idx:
            st.write(f"- **{project_names[i]}** — Cost: {costs[i]}M EGP | Utility: {utility[i]}")
 
        colors_dp = ['#2ecc71' if i in selected_idx else '#95a5a6'
                     for i in range(len(project_names))]
        fig_dp = go.Figure(go.Bar(
            x=project_names, y=utility,
            marker_color=colors_dp,
            text=utility, textposition='outside',
        ))
        fig_dp.update_layout(
            title="Utility Scores — Green = Selected",
            yaxis_title='Utility Score',
            xaxis_tickangle=-30,
            height=350
        )
        st.plotly_chart(fig_dp, width="stretch")
 
 
# TAB 4 — ML TRAFFIC PREDICTION
with tab4:
    st.header("ML-Based Traffic Congestion Prediction")
    st.write("Three scikit-learn models trained on Traffic_Flow_Data_Patterns.csv to forecast congestion.")
 
    # Model Performance
    st.subheader("Model Performance Comparison")
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for idx, (name, res) in enumerate(ml_results.items()):
        cols[idx].metric(name, f"R² = {res['r2']:.3f}", f"MAE = {res['mae']:.0f} veh/h")
 
    best_name = max(ml_results, key=lambda k: ml_results[k]['r2'])
    st.success(f"Best Model: **{best_name}** (R² = {ml_results[best_name]['r2']:.4f})")
 
    # Predicted vs Actual chart
    best = ml_results[best_name]
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=best['y_test'], y=best['preds'], mode='markers',
        marker=dict(size=7, opacity=0.7), name='Predictions'
    ))
    mn, mx = float(min(best['y_test'])), float(max(best['y_test']))
    fig_pred.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode='lines',
        line=dict(color='red', dash='dash'), name='Perfect Prediction'
    ))
    fig_pred.update_layout(
        title=f"{best_name} — Predicted vs Actual",
        xaxis_title='Actual (veh/h)', yaxis_title='Predicted (veh/h)',
        height=350
    )
    st.plotly_chart(fig_pred, width="stretch")
 
    # Heatmap
    st.subheader("Traffic Volume Heatmap")
    time_cols = ['Morning Peak(veh/h)', 'Afternoon (veh/h)', 'Evening Peak(veh/h)', 'Night(veh/h)']
    fig_heat = px.imshow(
        traffic_df.set_index('RoadID')[time_cols],
        labels=dict(x="Time Period", y="Road ID", color="Vehicles/h"),
        color_continuous_scale='RdYlGn_r', aspect='auto',
        title='Congestion Heatmap — All Roads'
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, width="stretch")
 
    # Interactive Predictor
    st.subheader("Predict Traffic for a Specific Road & Time")
    p1, p2 = st.columns(2)
    with p1:
        selected_road = st.selectbox("Select Road", traffic_df['RoadID'].tolist())
        time_of_day   = st.select_slider(
            "Time of Day",
            options=['Morning Peak (8am)', 'Afternoon (1pm)', 'Evening Peak (6pm)', 'Night (11pm)']
        )
    with p2:
        selected_model_name = st.selectbox("ML Model", list(ml_results.keys()))
 
    time_hour_map = {
        'Morning Peak (8am)': (0, 8),
        'Afternoon (1pm)':    (1, 13),
        'Evening Peak (6pm)': (2, 18),
        'Night (11pm)':       (3, 23),
    }
    t_idx, t_hour = time_hour_map[time_of_day]
 
    parts = str(selected_road).replace('F', '9').split('_')
    try:
        n1, n2 = int(parts[0]), int(parts[1])
    except:
        n1, n2 = 5, 5
 
    road_row   = traffic_df[traffic_df['RoadID'] == selected_road].iloc[0]
    road_avg   = road_row[time_cols].mean()
    sel_res    = ml_results[selected_model_name]
    X_new      = sel_res['scaler'].transform([[t_hour, t_idx, n1, n2, road_avg]])
    pred_val   = sel_res['model'].predict(X_new)[0]
    actual_val = road_row[time_cols[t_idx]]
 
    if pred_val > 2800:
        cong_label = "HIGH CONGESTION"
    elif pred_val > 1800:
        cong_label = "MODERATE"
    else:
        cong_label = "LOW CONGESTION"
 
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("ML Prediction",      f"{pred_val:,.0f} veh/h")
    res_col2.metric("Historical Actual",  f"{actual_val:,.0f} veh/h")
    res_col3.metric("Congestion Status",  cong_label)
    res_col4.metric("Prediction Error",   f"±{abs(pred_val - actual_val):,.0f}")
 
    # Daily profile chart
    predicted_profile = []
    for ti, th in enumerate(time_hours):
        Xp = sel_res['scaler'].transform([[th, ti, n1, n2, road_avg]])
        predicted_profile.append(sel_res['model'].predict(Xp)[0])
 
    actual_profile = [road_row[c] for c in time_cols]
    time_labels    = ['Morning Peak', 'Afternoon', 'Evening Peak', 'Night']
 
    fig_profile = go.Figure()
    fig_profile.add_trace(go.Scatter(
        x=time_labels, y=actual_profile, name='Actual',
        mode='lines+markers', line=dict(width=3), marker=dict(size=10)
    ))
    fig_profile.add_trace(go.Scatter(
        x=time_labels, y=predicted_profile, name=f'Predicted ({selected_model_name})',
        mode='lines+markers', line=dict(width=3, dash='dash'), marker=dict(size=10)
    ))
    fig_profile.update_layout(
        title=f'Daily Traffic Profile — Road {selected_road}',
        yaxis_title='Volume (veh/h)', height=320
    )
    st.plotly_chart(fig_profile, width="stretch")
 
    with st.expander("Feature Importance (Random Forest)"):
        rf = ml_results['Random Forest']['model']
        fi = pd.Series(rf.feature_importances_, index=ml_features).sort_values(ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation='h',
            text=[f'{v:.3f}' for v in fi.values], textposition='outside'
        ))
        fig_fi.update_layout(title='Feature Importance', height=280, margin=dict(l=120))
        st.plotly_chart(fig_fi, width="stretch")
        st.markdown("""
- **road_avg** — Average volume across all time periods (strongest predictor)
- **time_idx / hour** — Time-of-day encoding (captures rush-hour patterns)
- **node1 / node2** — Road endpoint IDs (proxy for network centrality)
        """)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM HELPERS FOR NEW TABS
# ─────────────────────────────────────────────────────────────────────────────
 
import heapq
 
# ── A. Time-Varying Dijkstra ──────────────────────────────────────────────────
RUSH_MULTIPLIERS = {
    "Morning Peak (8am)":   {"peak": 2.5, "normal": 1.0},
    "Afternoon (1pm)":      {"peak": 1.2, "normal": 1.2},
    "Evening Peak (6pm)":   {"peak": 2.8, "normal": 1.0},
    "Night (11pm)":         {"peak": 0.6, "normal": 0.6},
}
 
# Roads that are congested in Cairo (high-traffic corridors simulated by IDs)
PEAK_ROADS = {"1_2", "2_1", "3_4", "4_3", "5_6", "6_5", "7_8", "8_7",
              "9_10", "10_9", "2_5", "5_2", "1_3", "3_1", "4_6", "6_4"}
 
 
def time_varying_dijkstra(G, source, target, time_label):
    """Modified Dijkstra with time-dependent edge weights."""
    mult = RUSH_MULTIPLIERS.get(time_label, {"peak": 1.0, "normal": 1.0})
    dist = {n: float('inf') for n in G.nodes}
    dist[source] = 0
    prev = {n: None for n in G.nodes}
    pq = [(0, source)]
    visited = set()
 
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, edata in G[u].items():
            base_w = edata.get('weight', 1)
            road_key = f"{u}_{v}"
            factor = mult["peak"] if road_key in PEAK_ROADS else mult["normal"]
            new_d = d + base_w * factor
            if new_d < dist[v]:
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(pq, (new_d, v))
 
    # Reconstruct path
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if path[0] != source:
        return [], float('inf')
    return path, dist[target]
 
 
# ── B. Greedy Traffic Signal Optimizer ───────────────────────────────────────
def greedy_signal_optimization(intersections):
    """
    Greedy approach: at each intersection, allocate green-light time
    proportionally to the lane with the highest current traffic volume.
    Always serves the busiest direction first (greedy local optimum).
    Returns allocation plan and total weighted wait reduction.
    """
    results = []
    total_cycle = 120  # seconds per cycle
    for inter in intersections:
        lanes = inter["lanes"]
        total_vol = sum(v for _, v in lanes)
        if total_vol == 0:
            alloc = [(name, total_cycle // len(lanes)) for name, _ in lanes]
        else:
            # Greedy: sort by volume descending, allocate proportionally
            sorted_lanes = sorted(lanes, key=lambda x: x[1], reverse=True)
            alloc = []
            for name, vol in sorted_lanes:
                green_time = round((vol / total_vol) * total_cycle)
                alloc.append((name, green_time, vol))
        results.append({"name": inter["name"], "allocation": alloc,
                         "total_vol": total_vol})
    return results
 
 
# ── C. Public Transit Scheduling DP ──────────────────────────────────────────
def transit_scheduling_dp(routes, vehicles, time_slots):
    """
    DP for optimal vehicle scheduling across transit routes.
    State: dp[v][t] = max passengers served using v vehicles up to time slot t.
    Each route has a demand per slot and requires a fixed number of vehicles.
    """
    n_routes = len(routes)
    # dp[i][j] = max passengers served considering first i routes with j vehicles
    dp = [[0] * (vehicles + 1) for _ in range(n_routes + 1)]
    chosen = [[[] for _ in range(vehicles + 1)] for _ in range(n_routes + 1)]
 
    for i in range(1, n_routes + 1):
        route = routes[i - 1]
        req   = route["vehicles_needed"]
        gain  = route["daily_passengers"]
        for v in range(vehicles + 1):
            # Don't assign this route
            dp[i][v] = dp[i-1][v]
            chosen[i][v] = chosen[i-1][v][:]
            # Assign this route if enough vehicles
            if v >= req and dp[i-1][v - req] + gain > dp[i][v]:
                dp[i][v] = dp[i-1][v - req] + gain
                chosen[i][v] = chosen[i-1][v - req][:] + [route["name"]]
 
    return dp[n_routes][vehicles], chosen[n_routes][vehicles]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — TIME-VARYING SHORTEST PATH
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("⏱️ Time-Varying Route Planning (Modified Dijkstra)")
    st.info("""
    **Algorithm:** Modified Dijkstra that multiplies each edge weight by a
    congestion factor depending on the time of day.  
    Rush-hour roads (Cairo's main corridors) get a **2.5–2.8× penalty**,
    forcing the algorithm to discover alternative routes that are faster
    despite being longer in distance.
    """)
 
    if options:
        c1, c2, c3 = st.columns(3)
        tv_src  = c1.selectbox("Origin",      list(options.keys()), index=0,  key="tv_src")
        tv_dst  = c2.selectbox("Destination", list(options.keys()),
                               index=min(4, len(options)-1), key="tv_dst")
        tv_time = c3.selectbox("Time of Day", list(RUSH_MULTIPLIERS.keys()), key="tv_time")
 
        st.markdown("#### Rush-Hour Multipliers")
        mult_info = RUSH_MULTIPLIERS[tv_time]
        mi1, mi2 = st.columns(2)
        mi1.metric("Peak-Road Factor",   f"×{mult_info['peak']}")
        mi2.metric("Normal-Road Factor", f"×{mult_info['normal']}")
 
        if st.button("🔍 Find Time-Aware Route", key="tv_run"):
            s_id_tv = options[tv_src]
            e_id_tv = options[tv_dst]
 
            # Static Dijkstra (baseline)
            try:
                t0 = time.perf_counter()
                static_path = nx.shortest_path(cairo_network, s_id_tv, e_id_tv, weight='weight')
                static_time_ms = (time.perf_counter() - t0) * 1000
                static_dist = nx.path_weight(cairo_network, static_path, 'weight')
            except Exception:
                static_path, static_dist, static_time_ms = [], 0, 0
 
            # Time-varying Dijkstra
            t0 = time.perf_counter()
            tv_path, tv_weighted_dist = time_varying_dijkstra(
                cairo_network, s_id_tv, e_id_tv, tv_time)
            tv_time_ms = (time.perf_counter() - t0) * 1000
            tv_real_dist = (nx.path_weight(cairo_network, tv_path, 'weight')
                            if tv_path else 0)
 
            st.subheader("Comparison: Static vs Time-Aware")
            col_s, col_t = st.columns(2)
            with col_s:
                st.markdown("**Static Dijkstra** (ignores time of day)")
                st.metric("Distance",   f"{static_dist:.2f} km")
                st.metric("Nodes",      len(static_path))
                st.metric("Calc Time",  f"{static_time_ms:.3f} ms")
                if static_path:
                    components.html(generate_map_html(cairo_network, static_path),
                                    height=430)
            with col_t:
                st.markdown(f"**Time-Aware Dijkstra** ({tv_time})")
                st.metric("Real Distance",     f"{tv_real_dist:.2f} km")
                st.metric("Weighted Cost",     f"{tv_weighted_dist:.2f}")
                st.metric("Nodes",             len(tv_path))
                st.metric("Calc Time",         f"{tv_time_ms:.3f} ms")
                if tv_path:
                    components.html(generate_map_html(cairo_network, tv_path),
                                    height=430)
 
            if static_path and tv_path and static_path != tv_path:
                st.warning(
                    f"⚠️ The time-aware algorithm chose a **different route** "
                    f"to avoid congestion — real distance differs by "
                    f"{abs(tv_real_dist - static_dist):.2f} km.")
            elif static_path and tv_path:
                st.success("✅ Same route — no significant congestion penalty on this corridor.")
 
            with st.expander("📖 Algorithm Details"):
                st.markdown(f"""
**Time-Varying Dijkstra — Complexity: O((V + E) log V)**
 
For each edge *(u, v)* with base weight *w*:
```
effective_weight = w × congestion_factor(u, v, time_period)
```
 
| Time Period | Peak-Road Factor | Reason |
|---|---|---|
| Morning Peak (8am) | ×2.5 | Inbound commuter traffic |
| Afternoon (1pm) | ×1.2 | Moderate midday flow |
| Evening Peak (6pm) | ×2.8 | Worst congestion of the day |
| Night (11pm) | ×0.6 | Free-flowing roads |
 
The algorithm correctly routes vehicles around congested corridors even
when the alternate path is physically longer.
                """)
    else:
        st.warning("Data not found. Please check the 'csvFiles' directory.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — GREEDY TRAFFIC SIGNAL + EMERGENCY PREEMPTION
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.header("🚦 Greedy Traffic Signal Optimization & Emergency Preemption")
    st.info("""
    **Algorithm:** Greedy approach — at each intersection, sort lanes by current
    traffic volume (descending) and allocate green-light time proportionally.
    The busiest lane is always served first (greedy local optimum).  
    An **emergency preemption** mode overrides normal scheduling to clear the
    path for emergency vehicles using a priority queue.
    """)
 
    # ── Normal Signal Optimization ──
    st.subheader("Part A — Normal Signal Optimization")
 
    INTERSECTIONS = [
        {"name": "Tahrir Square",       "lanes": [("North↕", 3200), ("South↕", 2800), ("East↔", 1500), ("West↔", 1200)]},
        {"name": "Ramses Square",       "lanes": [("North↕", 2900), ("South↕", 2600), ("East↔", 3100), ("West↔", 2400)]},
        {"name": "Nasr City Junction",  "lanes": [("North↕", 1800), ("South↕", 2100), ("East↔", 2700), ("West↔", 1600)]},
        {"name": "Giza Square",         "lanes": [("North↕", 2200), ("South↕", 1900), ("East↔", 2500), ("West↔", 2000)]},
        {"name": "Heliopolis Cross",    "lanes": [("North↕", 1400), ("South↕", 1600), ("East↔", 1900), ("West↔", 1300)]},
    ]
 
    traffic_scale = st.slider(
        "Simulate Traffic Level (% of peak)", 40, 150, 100, key="sig_scale")
    scaled_inters = []
    for inter in INTERSECTIONS:
        scaled_lanes = [(name, int(vol * traffic_scale / 100))
                        for name, vol in inter["lanes"]]
        scaled_inters.append({"name": inter["name"], "lanes": scaled_lanes})
 
    if st.button("🟢 Run Greedy Signal Optimizer", key="sig_run"):
        plan = greedy_signal_optimization(scaled_inters)
 
        for result in plan:
            with st.expander(f"📍 {result['name']} — Total volume: {result['total_vol']:,} veh/h"):
                alloc_df = pd.DataFrame(result["allocation"],
                                        columns=["Lane", "Green Time (s)", "Volume (veh/h)"])
                st.dataframe(alloc_df, width="stretch")
 
                fig_sig = go.Figure(go.Bar(
                    x=[r[0] for r in result["allocation"]],
                    y=[r[1] for r in result["allocation"]],
                    marker_color=["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"],
                    text=[f"{r[1]}s" for r in result["allocation"]],
                    textposition="outside",
                ))
                fig_sig.update_layout(
                    title=f"Green Time Allocation — {result['name']}",
                    yaxis_title="Green Time (seconds)",
                    height=280,
                )
                st.plotly_chart(fig_sig, width="stretch")
 
    # ── Emergency Preemption ──
    st.markdown("---")
    st.subheader("Part B — Emergency Vehicle Preemption (Priority Queue)")
    st.markdown("""
    When an emergency vehicle is detected, a **max-priority queue** immediately
    re-orders intersection phases. The vehicle's direction receives 100 % of
    the green phase. All other lanes are held until the vehicle clears.
    """)
 
    EMERGENCY_ROUTES = {
        "Ambulance → Kasr Al-Ainy Hospital": ["Tahrir Square", "Ramses Square"],
        "Fire Truck → Nasr City Station":    ["Nasr City Junction", "Heliopolis Cross"],
        "Police → Giza HQ":                 ["Giza Square", "Tahrir Square"],
    }
 
    em_vehicle   = st.selectbox("Select Emergency Vehicle Route",
                                list(EMERGENCY_ROUTES.keys()), key="em_veh")
    em_direction = st.selectbox("Vehicle Approach Direction",
                                ["North↕", "South↕", "East↔", "West↔"], key="em_dir")
 
    if st.button("🚨 Activate Emergency Preemption", key="em_run"):
        affected = EMERGENCY_ROUTES[em_vehicle]
        st.error(f"🚨 EMERGENCY PREEMPTION ACTIVE — {em_vehicle}")
 
        for iname in affected:
            inter = next(i for i in INTERSECTIONS if i["name"] == iname)
            st.markdown(f"**{iname}**")
            pq_lanes = sorted(inter["lanes"], key=lambda x: x[1], reverse=True)
            cols_em = st.columns(len(pq_lanes))
            for idx, (lane, vol) in enumerate(pq_lanes):
                is_em = lane == em_direction
                cols_em[idx].metric(
                    label=lane,
                    value="🟢 100s GREEN" if is_em else "🔴 0s HELD",
                    delta="Emergency clearance" if is_em else "Preempted",
                    delta_color="normal" if is_em else "inverse",
                )
        st.success("✅ Emergency vehicle cleared. Normal signal plan resumes after passage.")
 
    with st.expander("📖 Greedy Analysis — Optimal vs Suboptimal Cases"):
        st.markdown("""
**When Greedy is Optimal:**
- Single-intersection, independent scheduling — proportional allocation minimises
  average wait time (provably optimal for M/D/1 queues).
- Low-variance traffic (all lanes have similar volumes) — greedy and optimal
  solutions coincide.
 
**When Greedy is Suboptimal:**
- **Correlated intersections:** Greedy ignores downstream queues. Serving the
  busiest lane at Tahrir may flood Ramses Square if the downstream signal
  isn't coordinated.
- **Starvation:** A lightly-loaded lane (e.g. 200 veh/h) may receive only
  7–8 s of green per cycle, causing pedestrian and side-street starvation.
- **Emergency aftermath:** After preemption, the held lanes have built up
  significant queues. Greedy resumes its normal proportional plan, which
  may be suboptimal for dissipating the accumulated backlog.
 
**Complexity:** O(L log L) per intersection, where L = number of lanes.
        """)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — PUBLIC TRANSIT SCHEDULING DP
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.header("🚌 Public Transit Scheduling — Dynamic Programming")
    st.info("""
    **Algorithm:** 0/1 Knapsack-style DP over transit routes.  
    Each route has a *vehicle requirement* and a *daily passenger demand*.  
    Given a fixed fleet size, the DP finds the assignment that **maximises
    total daily passengers served** — the same principle used by Cairo's
    public transport authority to allocate buses and metro trains.
    """)
 
    TRANSIT_ROUTES = [
        {"name": "Metro Line 1 (Helwan–New Marg)", "vehicles_needed": 12, "daily_passengers": 850_000},
        {"name": "Metro Line 2 (Shubra–Giza)",     "vehicles_needed": 10, "daily_passengers": 720_000},
        {"name": "Metro Line 3 (Airport–Imbaba)",  "vehicles_needed": 8,  "daily_passengers": 540_000},
        {"name": "Bus Rapid Transit — Ring Road",  "vehicles_needed": 6,  "daily_passengers": 180_000},
        {"name": "Bus Route — Tahrir–Nasr City",   "vehicles_needed": 4,  "daily_passengers": 95_000},
        {"name": "Bus Route — Giza–6 October",     "vehicles_needed": 5,  "daily_passengers": 110_000},
        {"name": "Microbus — Downtown Cairo",      "vehicles_needed": 3,  "daily_passengers": 60_000},
        {"name": "Minibus — Heliopolis Circle",    "vehicles_needed": 2,  "daily_passengers": 40_000},
        {"name": "Bus Route — Maadi–Zamalek",      "vehicles_needed": 4,  "daily_passengers": 85_000},
        {"name": "Night Bus — Airport Shuttle",    "vehicles_needed": 2,  "daily_passengers": 25_000},
    ]
 
    st.subheader("Available Transit Routes")
    routes_df = pd.DataFrame([{
        "Route": r["name"],
        "Vehicles Needed": r["vehicles_needed"],
        "Daily Passengers": f'{r["daily_passengers"]:,}',
        "Passengers/Vehicle": f'{r["daily_passengers"] // r["vehicles_needed"]:,}'
    } for r in TRANSIT_ROUTES])
    st.dataframe(routes_df, width="stretch")
 
    total_req = sum(r["vehicles_needed"] for r in TRANSIT_ROUTES)
    st.markdown(f"*Total vehicles needed to operate all routes: **{total_req}***")
 
    fleet_size = st.slider(
        "Available Fleet Size (vehicles)", 10, total_req, 30, key="fleet_slider")
 
    time_slots = st.multiselect(
        "Active Time Slots",
        ["Morning Peak", "Afternoon", "Evening Peak", "Night"],
        default=["Morning Peak", "Evening Peak"],
        key="ts_slots"
    )
 
    if st.button("🚌 Optimise Fleet Allocation", key="dp_transit_run"):
        best_passengers, chosen_routes = transit_scheduling_dp(
            TRANSIT_ROUTES, fleet_size, time_slots)
        used_vehicles = sum(r["vehicles_needed"]
                            for r in TRANSIT_ROUTES if r["name"] in chosen_routes)
        total_possible = sum(r["daily_passengers"] for r in TRANSIT_ROUTES)
        coverage_pct = best_passengers / total_possible * 100
 
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Passengers Served", f"{best_passengers:,}")
        m2.metric("Fleet Used",            f"{used_vehicles} / {fleet_size}")
        m3.metric("Routes Activated",      len(chosen_routes))
        m4.metric("Network Coverage",      f"{coverage_pct:.1f}%")
 
        st.subheader("✅ Activated Routes")
        act_data = []
        for r in TRANSIT_ROUTES:
            activated = r["name"] in chosen_routes
            act_data.append({
                "Route": r["name"],
                "Status": "✅ Active" if activated else "❌ Idle",
                "Vehicles": r["vehicles_needed"] if activated else 0,
                "Passengers": f'{r["daily_passengers"]:,}' if activated else "—",
            })
        act_df = pd.DataFrame(act_data)
        st.dataframe(act_df, width="stretch")
 
        # Bar chart — activated vs idle
        fig_transit = go.Figure()
        for r in TRANSIT_ROUTES:
            activated = r["name"] in chosen_routes
            fig_transit.add_trace(go.Bar(
                name=r["name"],
                x=["Daily Passengers"],
                y=[r["daily_passengers"]],
                marker_color="#2ecc71" if activated else "#95a5a6",
                showlegend=True,
            ))
        fig_transit.update_layout(
            barmode="stack",
            title=f"Passenger Coverage — Fleet of {fleet_size} vehicles",
            yaxis_title="Daily Passengers",
            height=380,
            legend=dict(orientation="h", y=-0.4),
        )
        st.plotly_chart(fig_transit, width="stretch")
 
        # DP table visualisation (first 6 routes, first 20 vehicle slots)
        with st.expander("📊 DP Table (first 6 routes × first 20 vehicle slots)"):
            n_show, v_show = min(6, len(TRANSIT_ROUTES)), min(20, fleet_size + 1)
            dp_vis = [[0] * v_show for _ in range(n_show + 1)]
            for i in range(1, n_show + 1):
                r = TRANSIT_ROUTES[i - 1]
                req, gain = r["vehicles_needed"], r["daily_passengers"]
                for v in range(v_show):
                    dp_vis[i][v] = dp_vis[i-1][v]
                    if v >= req and dp_vis[i-1][v - req] + gain > dp_vis[i][v]:
                        dp_vis[i][v] = dp_vis[i-1][v - req] + gain
            dp_df = pd.DataFrame(
                dp_vis[1:],
                index=[r["name"][:30] for r in TRANSIT_ROUTES[:n_show]],
                columns=[f"v={j}" for j in range(v_show)],
            )
            st.dataframe(dp_df.map(lambda x: f"{x:,}"), width="stretch")
            st.caption("Each cell = max passengers served using that many vehicles "
                       "considering routes up to that row.")
 
        with st.expander("📖 Algorithm Details & Complexity"):
            st.markdown(f"""
**Transit Scheduling DP — Complexity: O(R × V)**
 
Where **R** = number of routes ({len(TRANSIT_ROUTES)}) and **V** = fleet size ({fleet_size}).
 
**Recurrence:**
```
dp[i][v] = max(
    dp[i-1][v],                              # skip route i
    dp[i-1][v - req_i] + passengers_i        # activate route i
)
```
This is equivalent to the 0/1 Knapsack where:
- *Weight* = vehicles needed per route
- *Value*  = daily passengers served
- *Capacity* = total available fleet
 
**Why DP and not Greedy?**  
Sorting routes by *passengers/vehicle* (greedy) would activate the Airport
Shuttle early but miss the high-volume Metro Line 1 that requires 12 vehicles.
DP guarantees the global optimum by exploring all valid combinations.
            """)