from flask import Flask, request, jsonify
from predict import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    features = data["features"]
    result = predict(features)
    labels = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return jsonify({"prediction": labels[result]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)