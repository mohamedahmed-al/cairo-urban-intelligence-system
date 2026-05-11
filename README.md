# 🏙️ Cairo Urban Intelligence System

An interactive web application that applies **classic and modern algorithms** to real urban planning challenges in Cairo — from shortest-path routing to ML-based traffic prediction.

Built with **Python + Streamlit**, fully containerized with **Docker**, and deployable to the cloud in minutes.

---

## 🌐 Live Demo

> 🔗 **[cairo-urban-intelligence.onrender.com](https://cairo-urban-intelligence.onrender.com)**  
> *(First load may take ~30 seconds on the free tier)*

---

## 📸 Preview

| Algorithm Race | Traffic Prediction |
|---|---|
| Dijkstra vs A* side-by-side with interactive graph | ML model forecasting congestion by time of day |

---

## ✨ Features

### ⚔️ Tab 1 — Algorithm Race (Dijkstra vs A*)
- Select any two locations in Cairo and race both algorithms
- Side-by-side interactive network graphs
- Real-time calculation time comparison in milliseconds

### 🏗️ Tab 2 — Infrastructure Planning (MST)
- Minimum Spanning Tree to find the optimal road network
- Minimizes total construction cost while connecting all districts
- Visual graph of the recommended infrastructure layout

### 📊 Tab 3 — Budget Optimization (0/1 Knapsack DP)
- Allocate a limited urban budget across city projects
- Dynamic Programming approach to maximize impact per pound spent
- DP table visualization showing decision process

### 🤖 Tab 4 — ML Traffic Congestion Prediction
- Trained on Cairo traffic flow data (morning, afternoon, evening, night peaks)
- Models: Random Forest, Gradient Boosting, Linear Regression
- Feature importance charts and model accuracy metrics (MAE, R²)

### ⏱️ Tab 5 — Time-Varying Route Planning
- Modified Dijkstra that accounts for time-of-day congestion
- Compares standard shortest path vs congestion-aware route
- Highlights when the algorithm picks a longer but faster route

### 🚦 Tab 6 — Traffic Signal Optimization & Emergency Preemption
- Greedy algorithm to optimize signal timing across intersections
- Emergency vehicle preemption simulation
- Before/after congestion score comparison

### 🚌 Tab 7 — Public Transit Scheduling (DP)
- Dynamic Programming to optimize bus/metro scheduling
- Maximize coverage with limited fleet size
- Route allocation table with demand satisfaction metrics

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Graph Algorithms | NetworkX |
| ML Models | Scikit-learn (Random Forest, Gradient Boosting, Linear Regression) |
| Visualization | Plotly, PyVis |
| Data | Pandas, NumPy |
| Containerization | Docker |
| Deployment | Render |

---

## 📂 Project Structure

```
cairo-urban-intelligence-system/
│
├── app.py                          # Main Streamlit application
├── Dockerfile                      # Docker container config
├── render.yaml                     # Render deployment config
├── requirements.txt                # Python dependencies
├── README.md
│
└── csvFiles/
    ├── Geographic_Data(Neighborhoods_and_Districts).csv
    ├── Geographic_Data(Important_Facilities).csv
    ├── Road_Network_Data(Existing_Roads).csv
    ├── Road_Network_Data(Potential_New_Roads).csv
    ├── Traffic_Flow_Data_Patterns.csv
    ├── Public_Transportation_Data(current_bus_routes).csv
    ├── Public_Transportation_Data(current_metro_lines).csv
    └── Public_Transportation_Data(demand).csv
```

---

## 🚀 Run Locally

### Option 1 — Docker (Recommended)

```bash
# Build
docker build -t cairo-urban-intelligence .

# Run
docker run -p 8501:8501 cairo-urban-intelligence
```

Then open: **http://localhost:8501**

### Option 2 — Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## ☁️ Streamlit

   ```
   https://cairo-urban-intelligence-system-xblrywwvyotuecrnd75vdz.streamlit.app/
   ```

---

## 📊 Data Sources

All data is synthetic but geographically accurate, representing:
- **22 neighborhoods and districts** across Greater Cairo
- **15 important facilities** (airports, hospitals, universities, stations)
- **Road network** with real distance estimates in km
- **Traffic flow patterns** across 4 daily time periods

---

## 🧠 Algorithms Used

| Algorithm | Application |
|---|---|
| Dijkstra | Classic shortest path routing |
| A* | Heuristic-guided shortest path |
| Kruskal's MST | Optimal infrastructure planning |
| 0/1 Knapsack (DP) | Urban budget allocation |
| Modified Dijkstra | Time-aware congestion routing |
| Greedy Scheduling | Traffic signal optimization |
| DP Scheduling | Public transit fleet allocation |
| Random Forest | Traffic congestion prediction |
| Gradient Boosting | Traffic congestion prediction |

---

## 👨‍💻 Author

**Mohamed Ahmed**  
Algorithms Course Final Project — 2026

[LinkedIn]:https://www.linkedin.com/feed/update/urn:li:activity:7459550444005109760/

---

## 📜 License

This project is for educational purposes only.
