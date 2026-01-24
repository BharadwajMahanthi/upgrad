import requests
import json
import numpy as np

URL = "http://127.0.0.1:5000/predict_api"

# Create dummy payload with 19 features (PC0...PC18)
# Actual count depends on PCA variance retention
from src import config
n_features = len(config.get_feature_names())
print(f"Detected {n_features} features from config.")

dummy_features = np.random.randn(n_features).tolist()

payload = {
    "features": dummy_features
}

print(f"Testing API at: {URL}")
print(f"Payload (first 5 features): {dummy_features[:5]}...")

try:
    response = requests.post(URL, json=payload)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
        print("\n✅ API Test Passed!")
    else:
        print("Response Text:")
        print(response.text)
        print("\n❌ API Test Failed.")

except requests.exceptions.ConnectionError:
    print("\n❌ Could not connect to the server. Is app.py running?")
