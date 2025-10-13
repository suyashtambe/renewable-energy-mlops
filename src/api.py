from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load all trained models into memory
model_dir = "models"
models = {}

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        country_name = file.replace("_prophet.pkl", "").replace("_", " ")
        models[country_name] = joblib.load(os.path.join(model_dir, file))

@app.route("/")
def index():
    return "Renewable Energy Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    country = data.get("country")
    year = data.get("year")

    if country not in models:
        return jsonify({"error": f"No model available for {country}"}), 404

    try:
        model = models[country]
        # Prophet requires 'ds' column as datetime
        df = pd.DataFrame({"ds": [pd.to_datetime(f"{year}-01-01")]})
        forecast = model.predict(df)
        predicted_value = forecast["yhat"].values[0]
        return jsonify({"country": country, "year": year, "predicted_renewable_energy_share": float(predicted_value)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
