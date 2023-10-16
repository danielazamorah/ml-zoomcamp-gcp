import json

import pandas as pd
import joblib

from flask import Flask, request, abort, jsonify

app = Flask('churn')
@app.route('/predict', methods=['POST'])
def predict():
    # READ MODEL:
    model_pipeline = joblib.load('model.joblib')

    # READ DATA:
    customer = request.get_json()
    customer_pd = pd.json_normalize(customer)

    # MODEL PREDICTION:
    y_pred = model_pipeline.predict_proba(customer_pd)[:,1]

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(y_pred >= 0.5),
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
