# Thunderstorm-Forecast-Project

# ⛈️ Thunderstorm Forecasting with MLflow Tracking

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23150458.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-%2302569B.svg?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=MLflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end MLOps machine learning project designed to forecast severe atmospheric thunderstorm events using thermodynamic indicators and meteorological stability metrics. The system features modular pipeline architecture, dynamic experiment tracking with **MLflow**, and an interactive decision-support web portal built with **Streamlit**.

---

## 📌 Project Overview

Thunderstorms and severe convective activity pose significant risks to aviation, infrastructure, and public safety. Standard weather predictions can fail to capture non-linear atmospheric instabilities. This project solves that challenge by combining thermodynamic parameters—such as **CAPE (Convective Available Potential Energy)**, **K-Index**, **Dew Point Depression**, and **Wind Shear**—into a predictive MLOps pipeline.

### Core Capabilities
* **Thermodynamic Feature Engineering**: Automatically derives critical physical predictors like dew point depression, low-level moisture availability flags, and CAPE-shear kinematic interactions.
* **Multi-Model Training & Evaluation**: Trains and evaluates ensemble techniques (**Random Forest**, **XGBoost**, and **LightGBM**).
* **MLflow Experiment Tracking**: Logs hyperparameters, evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC), model binaries, and confusion matrix visual artifacts.
* **Streamlit Forecasting App**: Interactive web GUI allowing meteorologists and users to adjust atmospheric parameters dynamically and observe real-time predictions.

---

## 🏗️ Project Architecture & Directory Structure

```text
thunderstorm-forecasting-project/
├── data/
│   ├── generate_data.py          # Synthetic thermodynamic dataset generator
│   └── thunderstorm_data.csv     # Generated atmospheric readings
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py    # Domain-specific feature transformations
│   ├── train.py                  # Model training pipeline & MLflow experiment logger
│   └── app.py                    # Streamlit web dashboard
├── .gitignore                    # Environment & artifact exclusions
├── requirements.txt              # Project package dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation


**⚙️ Tech Stack**
**Language**: Python 3.11+

**Machine Learning**: Scikit-Learn, XGBoost, LightGBM

**Data Processing & Analytics**: Pandas, NumPy

**MLOps & Experiment Tracking**: MLflow

**Data Visualization**: Seaborn, Matplotlib

**Web UI Framework**: Streamlit


## **🏃 Execution Workflow**
Execute the pipeline in the following sequence:

Step 1: Generate Atmospheric Dataset
Construct the thermodynamic observation dataset:

