# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("my_ai_model")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    x = np.array(data["hours"]).reshape(-1,1)
    pred = model.predict(x)
    results = [float(p) for p in pred.flatten()]
    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(debug=True)