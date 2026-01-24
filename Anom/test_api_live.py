import json
import urllib.request
import urllib.error
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

from src.data.load_data import load_and_clean_data
from src.config import RAW_DATA_PATH

def test_api():
    print("Loading data to select a sample...")
    try:
        # Load data using our robust function
        df = load_and_clean_data(RAW_DATA_PATH)
        
        # Prepare a Normal sample (y=0)
        normal_sample = df[df['y'] == 0].iloc[0]
        # Prepare an Anomaly sample (y=1) if available
        anomaly_sample = df[df['y'] == 1].iloc[0] if 1 in df['y'].values else None
        
        # Drop target 'y' to get features only
        normal_features = normal_sample.drop('y').tolist()
        anomaly_features = anomaly_sample.drop('y').tolist() if anomaly_sample is not None else None
        
        API_URL = "http://127.0.0.1:5000/predict_api"
        
        # Test 1: Normal Sample
        print("\n--- Test 1: Normal Sample ---")
        payload = {"data": normal_features}
        send_request(API_URL, payload)
        
        # Test 2: Anomaly Sample
        if anomaly_features:
            print("\n--- Test 2: Anomaly Sample ---")
            payload = {"data": anomaly_features}
            send_request(API_URL, payload)
            
    except Exception as e:
        print(f"Test Setup Failed: {e}")

def send_request(url, payload):
    try:
        json_data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
            print(f"Status Code: {response.getcode()}")
            print(f"Response: {json.dumps(result, indent=2)}")
            
    except urllib.error.HTTPError as e:
        print(f"API Request Failed: HTTP Error {e.code}: {e.reason}")
        error_body = e.read().decode('utf-8')
        print(f"Error Body: {error_body}")
    except urllib.error.URLError as e:
        print(f"Connection Failed: {e}")
        print("Ensure the Flask app is running on port 5000!")

if __name__ == "__main__":
    test_api()
