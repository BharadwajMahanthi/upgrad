import requests
import numpy as np

URL = "http://127.0.0.1:5000/predict"

# 1. Define the same configuration as frontend
FEATURE_CONFIG = [
    { "label": "Credit Score", "min": 300, "max": 850,  "pMin": -150, "pMax": 150 },
    { "label": "Income",       "min": 20000, "max": 200000, "pMin": -200, "pMax": 300 },
    { "label": "Loan Amount",  "min": 1000, "max": 50000, "pMin": -750, "pMax": 1100 },
    { "label": "Balance",      "min": 0, "max": 100000, "pMin": -190, "pMax": 290 },
    { "label": "Employment",   "min": 0, "max": 40, "pMin": -540, "pMax": 400 },
    { "label": "DebtRatio",    "min": 0, "max": 100, "pMin": -105, "pMax": 70 },
    { "label": "OpenCredit",   "min": 0, "max": 30, "pMin": -320, "pMax": 430 },
    { "label": "Inquiries",    "min": 0, "max": 20, "pMin": -370, "pMax": 330 },
    { "label": "Marks",        "min": 0, "max": 10, "pMin": -40, "pMax": 40 },
    { "label": "CreditLimit",  "min": 1000, "max": 500000, "pMin": -340, "pMax": 370 },
    { "label": "Utilization",  "min": 0, "max": 100, "pMin": -32, "pMax": 34 },
    { "label": "PaymentHist",  "min": 0, "max": 100, "pMin": -67, "pMax": 60 },
    { "label": "History",      "min": 0, "max": 50, "pMin": -90, "pMax": 105 },
    { "label": "Term",         "min": 12, "max": 72, "pMin": -16, "pMax": 17 },
    { "label": "Rate",         "min": 2, "max": 30, "pMin": -150, "pMax": 130 },
    { "label": "Installment",  "min": 50, "max": 2000, "pMin": -50, "pMax": 56 },
    { "label": "HomeOwn",      "min": 0, "max": 10, "pMin": -17, "pMax": 16 },
    { "label": "ZipRisk",      "min": 0, "max": 100, "pMin": -25, "pMax": 24 },
    { "label": "Policy",       "min": 0, "max": 5, "pMin": -28, "pMax": 24 }
]

def scale_value(val, cfg):
    """Replicate JS scaleValue function"""
    uMin, uMax = cfg["min"], cfg["max"]
    mMin, mMax = cfg["pMin"], cfg["pMax"]
    
    pct = (val - uMin) / (uMax - uMin)
    return mMin + (pct * (mMax - mMin))

# 2. Simulate User Input (A "Safe" Profile)
user_inputs = [
    750,    # Credit Score (Good)
    85000,  # Income (High)
    15000,  # Loan Amount
    5000,   # Balance
    10,     # Employment
    20,     # Debt Ratio
    5,      # Open Credit
    0,      # Inquiries
    0,      # Marks
    50000,  # Limit
    10,     # Utilization (Low)
    100,    # Payment Hist (Perfect)
    15,     # History
    36,     # Term
    5,      # Rate (Low)
    400,    # Installment
    8,      # Home Own
    10,     # Zip Risk
    1       # Policy
]

print(f"Testing Flow with User Inputs: {user_inputs}")

# 3. Scale Inputs
scaled_inputs = []
for val, cfg in zip(user_inputs, FEATURE_CONFIG):
    scaled_inputs.append(scale_value(val, cfg))

print(f"\nScaled Inputs (PCA approx): {[round(x, 2) for x in scaled_inputs]}")

# 4. Submit to Backend
data_str = ",".join(map(str, scaled_inputs))
payload = {"data": data_str}

try:
    response = requests.post(URL, data=payload)
    
    if response.status_code == 200:
        print("\n✅ Server accepted data.")
        
        # Check for keywords in HTML response
        if "prediction" in response.text.lower():
             print("✅ Prediction returned in HTML.")
             
        if "low risk" in response.text.lower():
            print("✅ Result: LOW RISK (As expected for good profile)")
        elif "high risk" in response.text.lower():
            print("⚠️ Result: HIGH RISK (Unexpected for this profile, but model might disagree)")
        else:
            print("❓ Result: Unknown (Check HTML output)")
            
    else:
        print(f"❌ Server Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Connection Error: {e}")
