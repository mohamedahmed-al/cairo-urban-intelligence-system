import streamlit as st
import pandas as pd
import networkx as nx
import math
import time
from pyvis.network import Network
import streamlit.components.v1 as components

# إعداد الصفحة وتنسيقها
st.set_page_config(page_title="Cairo Smart City Dashboard", layout="wide")

# --- 1. تحميل البيانات وبناء الرسم البياني (مع مراجعة المسارات الجديدة) ---
@st.cache_data
def load_and_build_graph():
    G = nx.Graph()
    id_to_name = {}
    
    # تحديث المسارات لتقرأ من مجلد csvFiles
    try:
        # تحميل المناطق
        df_n = pd.read_csv("csvFiles/Geographic_Data(Neighborhoods_and_Districts).csv")
        for _, row in df_n.iterrows():
            nid = str(row['ID'])
            id_to_name[nid] = row['Name']
            G.add_node(nid, name=row['Name'], label=row['Name'], 
                       group="Neighborhood", x=row['X-coordinate'], y=row['Y-coordinate'])

        # تحميل المرافق الهامة
        df_f = pd.read_csv("csvFilesGeographic_Data(Important_Facilities).csv")
        for _, row in df_f.iterrows():
            fid = str(row['ID'])
            id_to_name[fid] = row['Name']
            G.add_node(fid, name=row['Name'], label=row['Name'], 
                       group="Facility", x=row['X-coordinate'], y=row['Y-coordinate'])

        # تحميل الطرق الحالية
        df_r = pd.read_csv("csvFiles/Road_Network_Data(Existing_Roads).csv")
        for _, row in df_r.iterrows():
            G.add_edge(str(row['FromID']), str(row['TOID']), weight=row['Distance(km)'])
            
    except FileNotFoundError as e:
        st.error(f"خطأ: لم يتم العثور على الملف في المسار المحدده: {e}")
        
    return G, id_to_name

cairo_network, id_to_name = load_and_build_graph()

# --- 2. محرك الرسوم المتحركة (Pyvis) ---
def generate_animated_map(graph, path=None):
    net = Network(height="500px", width="100%", bgcolor="#f8f9fa", font_color="#343a40")
    
    for node, data in graph.nodes(data=True):
        is_in_path = path and node in path
        color = "#ff4b4b" if is_in_path else ("#007bff" if data['group'] == "Neighborhood" else "#28a745")
        size = 25 if is_in_path else 15
        net.add_node(node, label=data['name'], color=color, size=size)

    for u, v, data in graph.edges(data=True):
        is_path_edge = False
        if path:
            path_edges = list(zip(path, path[1:]))
            if (u, v) in path_edges or (v, u) in path_edges:
                is_path_edge = True
        
        width = 5 if is_path_edge else 1
        color = "#ff4b4b" if is_path_edge else "#ced4da"
        net.add_edge(u, v, width=width, color=color)

    net.set_options('{"physics": {"forceAtlas2Based": {"avoidOverlap": 1}, "solver": "forceAtlas2Based"}}')
    return net.generate_html()

# --- 3. واجهة المستخدم ---
st.title("🏙️ ذكاء تنقل القاهرة الكبرى")
st.markdown("---")

tab1, tab2 = st.tabs(["🚀 سباق الخوارزميات", "🏗️ تخطيط البنية التحتية"])

with tab1:
    st.header("مقارنة Dijkstra vs. A*")
    col_a, col_b = st.columns(2)
    options = {v: k for k, v in id_to_name.items()}
    
    start_point = col_a.selectbox("نقطة البداية", list(options.keys()))
    end_point = col_b.selectbox("نقطة النهاية", list(options.keys()), index=len(options)-1)
    
    s_id, e_id = options[start_point], options[end_point]

    if st.button("🏁 ابدأ السباق"):
        c1, c2 = st.columns(2)
        
        # خوارزمية Dijkstra
        with c1:
            st.subheader("Dijkstra (التكلفة الموحدة)")
            t0 = time.perf_counter()
            d_path = nx.shortest_path(cairo_network, s_id, e_id, weight='weight')
            t1 = time.perf_counter()
            st.write(f"⏱️ **{(t1-t0)*1000:.3f}ms** | 📏 **{nx.path_weight(cairo_network, d_path, 'weight'):.2f}km**")
            components.html(generate_animated_map(cairo_network, d_path), height=520)

        # خوارزمية A*
        with c2:
            st.subheader("A* (البحث الذكي)")
            def dist_heuristic(u, v):
                n1, n2 = cairo_network.nodes[u], cairo_network.nodes[v]
                return math.sqrt((n1['x']-n2['x'])**2 + (n1['y']-n2['y'])**2) * 111

            t0 = time.perf_counter()
            a_path = nx.astar_path(cairo_network, s_id, e_id, heuristic=dist_heuristic, weight='weight')
            t1 = time.perf_counter()
            st.write(f"⏱️ **{(t1-t0)*1000:.3f}ms** | 📏 **{nx.path_weight(cairo_network, a_path, 'weight'):.2f}km**")
            components.html(generate_animated_map(cairo_network, a_path), height=520)

with tab2:
    st.header("تخطيط البنية التحتية (MST)")
    if st.button("🏗️ إنشاء المخطط العام"):
        mst = nx.minimum_spanning_tree(cairo_network, weight='weight')
        components.html(generate_animated_map(mst), height=600)