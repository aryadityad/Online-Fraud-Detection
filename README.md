# 💳 Online Payments Fraud Detection using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Flask-2.0%2B-green?style=for-the-badge&logo=flask" alt="Flask Version">
  <img src="https://img.shields.io/badge/Scikit--learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn Version">
  <img src="https://img.shields.io/badge/XGBoost-1.5%2B-purple?style=for-the-badge&logo=xgboost" alt="XGBoost Version">
</p>

An end-to-end machine learning pipeline that detects fraudulent online transactions using real-world payment data. Built with Python, powered by XGBoost, and deployed through a lightweight Flask-based web interface for real-time predictions.

---

## ✨ Key Features

- 🔄 **End-to-End ML Workflow** — From data ingestion to deployment, everything is covered.
- ⚡ **Robust Classifier** — XGBoost model tuned to tackle class imbalance and deliver high accuracy.
- 🔬 **Advanced Preprocessing** — Feature engineering, normalization, and [SMOTE](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#SMOTE) for balancing.
- 🌐 **Live Web App** — Clean, interactive Flask interface for quick predictions.
- 📦 **Modular & Reproducible** — Well-structured repo with easy environment setup via `requirements.txt`.

---

## 🛠️ Tech Stack

| Technology         | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| 🐍 Python          | Core language for scripting, modeling, and backend logic                    |
| ⚗️ Flask           | Lightweight WSGI web framework for model deployment                         |
| 🧮 Pandas / NumPy   | Efficient data manipulation and vectorized computation                      |
| 📊 Scikit-learn    | Preprocessing, pipeline creation, and basic ML utilities                    |
| ⚖️ Imbalanced-learn | SMOTE-based techniques to address dataset imbalance                        |
| 🚀 XGBoost         | High-performance gradient boosting model for classification                 |
| 📂 Kaggle          | Source of the anonymized credit card transaction dataset                    |
| 🌐 HTML / CSS      | UI structure and styling for the Flask-based interface                      |

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/aryadityad/Online-Fraud-Detection.git
cd Online-Fraud-Detection
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
```

- **Windows**:
  ```powershell
  .\venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python fraud_model_trainer.py
```

This script handles data download, preprocessing, model training, and saves the final pipeline to `fraud_detection_pipeline.pkl`.

### 5. Launch the Web App

```bash
python app.py
```

---

## 🖥️ How to Use

1. Once the app is running, visit: **http://127.0.0.1:5000/**
2. Click **"Predict"** to go to the input form.
3. Enter transaction data and hit **Submit**.
4. The model will return: **"Legitimate"** or **"Fraudulent"**.

---

## 🎥 Demo

Watch the full project in action here:  
🔗 [View Demo Video](https://drive.google.com/drive/folders/1cG7smR201sOGpD88vv7-DWmZASbU5Tfv?usp=sharing)

---

## 🔮 Future Scope

- ☁️ **Cloud Deployment** — Host on Heroku, AWS, or GCP.
- 🔁 **CI/CD Pipeline** — Automate model retraining and deployment with new data.
- 🧠 **Deep Learning** — Explore RNNs/LSTMs to capture sequential transaction patterns.
- 📊 **UI Enhancements** — Add visual feedback like prediction probabilities and feature importance charts.

---

## 👤 Author

**Aryaditya Deshmukh**  
🎓 Computer Science Engineering Student  
💻 Passionate about machine learning, cybersecurity, and building impactful software.

<p align="left">
  <a href="https://github.com/aryadityad" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-aryadityad-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

---
