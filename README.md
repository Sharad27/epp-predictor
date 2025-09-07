# 🏦 EPP Predictor

A **Machine Learning + FastAPI project** that predicts whether a credit card customer is likely to opt for an **Easy Payment Plan (EPP)** in the next 3 months.  
The project is fully containerized using **Docker**, making it portable and ready for deployment.

---

## 🚀 Features
- Predicts probability of a customer opting for EPP.
- Exposes a web interface (via FastAPI) for inputs and predictions.
- Includes REST API with **Swagger Docs** (`/docs`).
- Dockerized for consistent deployment.
- Uses an ensemble of **Random Forest, XGBoost, and LightGBM** for improved accuracy.

---

## 🖼️ Screenshots
![Front-end UI](images/front-end.png)

---

## ⚙️ Tech Stack
- **Python 3.11**
- **FastAPI** (API framework)
- **scikit-learn, XGBoost, LightGBM** (ML models)
- **Pandas, NumPy** (data processing)
- **Docker** (containerization)

---

## 📂 Project Structure

New Project/
│
├── data/ # Raw or processed datasets
├── model/ # ML logic: data sourcing, model training, saved models
├── static/ # CSS, images
├── templates/ # HTML templates for FastAPI web UI
├── venv/ # Virtual environment (not needed in Docker/GitHub)
│
├── pycache/ # Python cache files
├── config.py # Config variables (paths, constants, etc.)
├── main.py # FastAPI app entry point
├── train.py # Model training script
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build instructions
└── README.md # Project documentation
