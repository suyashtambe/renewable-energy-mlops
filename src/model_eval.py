# src/model_eval.py
import os
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error, r2_score
from prophet import Prophet
import yaml
import mlflow
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    
def evaluate_models():
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    test_path = config["test_data"]
    target = config["target"]

    test_df = pd.read_csv(test_path)

    if "Country" not in test_df.columns:
        raise KeyError("Country column missing in test data!")

    results = {}
    mlflow.set_experiment("Renewable-Energy-MLOPS")

    for model_file in os.listdir("models"):
        if not model_file.endswith("_prophet.pkl"):
            continue

        country = model_file.replace("_prophet.pkl", "").replace("_", " ")
        model_path = os.path.join("models", model_file)

        print(f"\n Evaluating model for {country}...")

        model = joblib.load(model_path)

        test_country = test_df[test_df["Country"] == country]
        if test_country.empty:
            print(f"⚠️ Skipping {country} (no test data)")
            continue

        # Prepare Prophet input
        start_year = test_df["Year_diff"].min() + 2000
        test_country = test_country[["Year_diff", target]].rename(
            columns={"Year_diff": "ds", target: "y"}
        )
        test_country["ds"] = pd.to_datetime(test_country["ds"] + start_year, format="%Y")

        forecast = model.predict(test_country[["ds"]])
        y_pred = forecast["yhat"].values
        y_true = test_country["y"].values

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results[country] = {"MAE": mae, "R2": r2}

        with mlflow.start_run(run_name=f"{country}_Evaluation"):
            mlflow.log_param("country", country)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

        print(f" {country}: MAE={mae:.2f}, R²={r2:.2f}")

    # Save evaluation results for DVC
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n Saved metrics to metrics/eval_metrics.json")

if __name__ == "__main__":
    evaluate_models()
