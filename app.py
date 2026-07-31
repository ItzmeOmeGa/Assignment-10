from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/")
def home():
    return "Heart Disease Prediction API is Running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = np.array([
        data["age"],
        data["sex"],
        data["cp"],
        data["trestbps"],
        data["chol"],
        data["fbs"],
        data["restecg"],
        data["thalach"],
        data["exang"],
        data["oldpeak"],
        data["slope"],
        data["ca"],
        data["thal"]
    ]).reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)