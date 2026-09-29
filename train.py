import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from feature_engineering import engineer_features

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Storm', 'Storm'], 
                yticklabels=['No Storm', 'Storm'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    plot_path = f"data/cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def train_and_evaluate():
    # 1. Load and prepare data
    raw_df = pd.read_csv("data/thunderstorm_data.csv")
    processed_df = engineer_features(raw_df)
    
    X = processed_df.drop(columns=["thunderstorm_event"])
    y = processed_df["thunderstorm_event"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Configure MLflow Experiment
    mlflow.set_experiment("Thunderstorm_Forecasting_MLOps")
    
    models = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42),
            {"n_estimators": 120, "max_depth": 8, "criterion": "gini"}
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.03, random_state=42),
            {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.03}
        ),
        "LightGBM": (
            LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42),
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05}
        )
    }
    
    for model_name, (model_obj, params) in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            # Fit model
            model_obj.fit(X_train, y_train)
            preds = model_obj.predict(X_test)
            probs = model_obj.predict_proba(X_test)[:, 1]
            
            # Compute evaluation metrics
            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds))
            rec = float(recall_score(y_test, preds))
            f1 = float(f1_score(y_test, preds))
            auc = float(roc_auc_score(y_test, probs))
            
            # Log Parameters & Metrics
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            # Save and log confusion matrix artifact
            cm_plot = plot_confusion_matrix(y_test, preds, model_name)
            mlflow.log_artifact(cm_plot)
            
            # Log Model Artifact
            mlflow.sklearn.log_model(model_obj, artifact_path="model")  #type: ignore
            # Log Model Artifact using the universal models interface 

            # mlflow.models.log_model(model_obj, artifact_path="model")

            print(f"Logged [{model_name}] | Run ID: {run.info.run_id} | Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    train_and_evaluate()