# app.py

from flask import Flask, request, jsonify
from model import predict_sentiment

app = Flask(__name__)

@app.route("/")
def home():
    return "NLP API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    result = predict_sentiment(text)
    
    return jsonify({
        "input": text,
        "sentiment": result
    })

if __name__ == "__main__":
    app.run(debug=True)