import os
import re
import json

import pandas as pd
import joblib

from flask import Flask, request, abort, jsonify

# from google.cloud import storage

# PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
# HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
# AIP_STORAGE_URI = os.environ["AIP_STORAGE_URI"]  # Vertex AI sets this env with path to the model artifact

# MODEL_PATH = "model.joblib"

app = Flask('churn')

# def decode_gcs_url(url: str) -> Tuple[str, str, str]:
#     """
#         Split a google cloud storage path such as: gs://bucket_name/dir1/filename into
#         bucket and path after the bucket: bucket_name, dir1/filename
#         :param url: storage url
#         :return: bucket_name, blob
#         """
#     bucket = re.findall(r'gs://([^/]+)', url)[0]
#     blob = url.split('/', 3)[-1]
#     return bucket, blob

# def download_artifacts(artifacts_uri:str, local_path:str):
#     storage_client = storage.Client()
#     src_bucket, src_blob = decode_gcs_url(artifacts_uri)
#     source_bucket = storage_client.bucket(src_bucket)
#     source_blob = source_bucket.blob(src_blob)
#     source_blob.download_to_filename(local_path)

# def load_artifacts(artifacts_uri:str=AIP_STORAGE_URI):
#     model_uri = os.path.join(artifacts_uri, "model")
#     download_artifacts(model_uri, MODEL_PATH)

# Flask route for Liveness checks
@app.route(HEALTH_ROUTE, methods=['GET'])
def health_check():
    return "I am alive, 200"

@app.route(PREDICT_ROUTE, methods=['POST'])
def predict():
    # GET MODEL ARTIFACT TO LOCAL:
    load_artifacts()
    model_pipeline = joblib.load(MODEL_PATH)

    # READING INCOMING DATA:
    payload = json.loads(request.data)
    instances = payload["instances"]

    try:
        df_str = "\n".join(instances)
        instances = pd.read_json(df_str, lines=True)
    except Exception as e:
        abort(500, "Failed to score request.")

    #customer = request.get_json()
    #customer_pd = pd.json_normalize(customer)
    # MODEL PREDICTION:
    y_pred = model_pipeline.predict_proba(instances)[:,1]

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(y_pred >= 0.5),
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
