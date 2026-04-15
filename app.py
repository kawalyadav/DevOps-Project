from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "AI Model API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        prediction = model.predict([data])
        
        return jsonify({
            "input": data,
            "prediction": int(prediction[0]),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()
