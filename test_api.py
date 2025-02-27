import requests

API_URL = "http://127.0.0.1:5000/predict"

data = {
    "keypoints": [0.5, 0.2, 0.6, 0.3, 0.7, 0.4, 0.8, 0.5, 0.9, 0.6]
}

response = requests.post(API_URL, json=data)
print("Response:", response.json())
