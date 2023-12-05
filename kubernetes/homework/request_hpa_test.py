import requests
import time

url = "http://localhost:8080/predict"

client = {"job": "retired", "duration": 445, "poutcome": "success"}

while True:
    time.sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)
