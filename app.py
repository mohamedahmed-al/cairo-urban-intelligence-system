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
tab1, tab2, tab3, tab4 = st.tabs([
    "⚔️ Algorithm Race",
    "🏗️ Infrastructure (MST)",
    "📊 Budget Optimization (DP)",
    "🤖 ML Traffic Prediction"
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
                st.plotly_chart(fig_bar, use_container_width=True)

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
        }), use_container_width=True)

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
        st.plotly_chart(fig_dp, use_container_width=True)


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
    st.plotly_chart(fig_pred, use_container_width=True)

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
    st.plotly_chart(fig_heat, use_container_width=True)

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
    st.plotly_chart(fig_profile, use_container_width=True)

    with st.expander("Feature Importance (Random Forest)"):
        rf = ml_results['Random Forest']['model']
        fi = pd.Series(rf.feature_importances_, index=ml_features).sort_values(ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation='h',
            text=[f'{v:.3f}' for v in fi.values], textposition='outside'
        ))
        fig_fi.update_layout(title='Feature Importance', height=280, margin=dict(l=120))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("""
- **road_avg** — Average volume across all time periods (strongest predictor)
- **time_idx / hour** — Time-of-day encoding (captures rush-hour patterns)
- **node1 / node2** — Road endpoint IDs (proxy for network centrality)
        """)
