# src/model_eval.py

import pandas as pd
import joblib
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    model = joblib.load(config["model"]["path"])
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("📊 Evaluation Results:")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    evaluate_model()
