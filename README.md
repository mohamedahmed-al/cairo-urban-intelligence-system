# 🚀 Algo Final Project - Streamlit App

## 📌 Overview

This project is an interactive web application built using **Streamlit** to demonstrate and visualize algorithmic concepts and data analysis. The app provides a user-friendly interface for exploring datasets, running computations, and displaying results dynamically.

The application is fully containerized using **Docker**, making it easy to run on any system without worrying about dependencies or environment setup.

---

## 🎯 Features

* 📊 Interactive data visualization
* ⚡ Fast and lightweight UI using Streamlit
* 🧠 Algorithm demonstration and analysis
* 📁 CSV file support for input data
* 🐳 Docker support for easy deployment

---

## 🛠️ Tech Stack

* **Python 3**
* **Streamlit**
* **Pandas**
* **Scikit-learn** (if used)
* **Docker**

---

## 📂 Project Structure

```
AlgoFinalProject/
│── app.py
│── Dockerfile
│── requirements.txt
│── csvFiles/
│── README.md
```

---

## ⚙️ Installation & Run (Without Docker)

### 1. Clone the repository

```
git clone <your-repo-link>
cd AlgoFinalProject
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 🐳 Run Using Docker (Recommended)

### 1. Build Docker Image

```
docker build -t algo-app .
```

### 2. Run Container

```
docker run -p 8501:8501 algo-app
```

### 3. Open in Browser

```
http://localhost:8501
```

---

## ❗ Notes

* Make sure Docker is installed and running.
* Ensure the correct file name is used in the Dockerfile (`app.py`).
* If port 8501 is busy, you can map another port:

```
docker run -p 8502:8501 algo-app
```

---

## 📈 Future Improvements

* Add more algorithms and visualizations
* Improve UI/UX design
* Deploy to cloud (AWS / Azure / GCP)
* Add real-time data processing

---

## 👨‍💻 Author

Developed as part of an Algorithms course project.


---

---

## Demo Vedio

Link:
https://www.linkedin.com/posts/mohamed-ahmed-4b9572407_httpslnkdindwqrudxf-ugcPost-7457022670702800897-ZxOV?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAGfYM00BmGH4pHxBOe8cvHLYZwKIpdy3O6s


---


## 📜 License

This project is for educational purposes.
