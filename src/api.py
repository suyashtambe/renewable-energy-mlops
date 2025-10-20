import os
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model_dir = "models"

# Load all available models
models = {}
if os.path.exists(model_dir):
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            country = file.replace("_prophet.pkl", "").replace("_model.pkl", "")
            models[country] = joblib.load(os.path.join(model_dir, file))
else:
    os.makedirs(model_dir, exist_ok=True)

@app.route("/")
def home():
    return jsonify({"message": "Renewable Energy MLOps API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    country = data.get("country")
    year = data.get("year")

    if country not in models:
        return jsonify({"error": f"No model found for {country}"}), 400

    model = models[country]
    future_df = model.make_future_dataframe(periods=1, freq='Y')
    forecast = model.predict(future_df.tail(1))
    yhat = forecast['yhat'].values[0]

    return jsonify({
        "country": country,
        "predicted_consumption": round(float(yhat), 2),
        "year": year
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
