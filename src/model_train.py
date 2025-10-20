# src/model_train.py
import pandas as pd
import joblib
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
import yaml
import mlflow
import mlflow.sklearn
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        # reconfigure may not be available in some environments; ignore if it fails
        pass

def train_per_country_models():
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    train_path = config["train_data"]
    test_path = config["test_data"]
    target = config["target"]

    # Load processed data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "Country" not in train_df.columns:
        raise KeyError("Country column missing in processed CSV. Keep it in preprocessing.")

    countries = train_df["Country"].unique()
    os.makedirs("models", exist_ok=True)

    # Set MLflow experiment name
    mlflow.set_experiment("Renewable-Energy-Demand")

    for country in countries:
        print(f"\n Training Prophet model for {country}...")

        # Filter data for current country
        train_country = train_df[train_df["Country"] == country][["Year_diff", target]].rename(
            columns={"Year_diff": "ds", target: "y"}
        )
        test_country = test_df[test_df["Country"] == country][["Year_diff", target]].rename(
            columns={"Year_diff": "ds", target: "y"}
        )

        if train_country.empty or test_country.empty:
            print(f" Skipping {country} (no data)")
            continue

        # Convert Year_diff to datetime
        start_year = train_df["Year_diff"].min() + 2000  # Replace 2000 with earliest dataset year if known
        train_country["ds"] = pd.to_datetime(train_country["ds"] + start_year, format="%Y")
        test_country["ds"] = pd.to_datetime(test_country["ds"] + start_year, format="%Y")

        # Prophet training within MLflow tracking
        with mlflow.start_run(run_name=f"{country}_Prophet"):
            model = Prophet(yearly_seasonality=False, daily_seasonality=False)
            model.fit(train_country)

            # Predict on test
            future = test_country[["ds"]]
            forecast = model.predict(future)
            y_pred = forecast["yhat"].values
            y_true = test_country["y"].values

            # Metrics
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            print(f" {country} trained! MAE: {mae:.2f}, R²: {r2:.2f}")

            # Log metrics to MLflow
            mlflow.log_param("country", country)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # Save model locally
            model_path = f"models/{country.replace(' ', '_')}_prophet.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            print(f" Model saved: {model_path}")

if __name__ == "__main__":
    train_per_country_models()
