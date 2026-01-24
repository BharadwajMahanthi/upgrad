from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import json

# ------------------------------------------------------------
# Add project root to sys.path
# ------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src import config
from src.models.model_defs import LogisticRegressionTorch


# ------------------------------------------------------------
# Flask app init
# ------------------------------------------------------------
app = Flask(__name__)


# ------------------------------------------------------------
# Load feature names dynamically (NO hardcoding ✅)
# ------------------------------------------------------------
feature_names = config.get_feature_names(prefer_resampled=True)

if not feature_names:
    raise RuntimeError(
        "❌ Could not detect feature names automatically.\n"
        "Make sure at least one of these files exists:\n"
        f"- {config.X_RESAMPLED_FILE}\n"
        f"- {config.X_SCALED_FILE}\n"
        f"- {config.RAW_DATA_FILE}\n"
    )

input_dim = len(feature_names)
print(f"✅ Detected {input_dim} feature columns.")


# ------------------------------------------------------------
# Load Torch model (.pt) + threshold metadata
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prefer BEST_TORCH_MODEL_FILE if you used the updated config.py
model_path = getattr(config, "BEST_TORCH_MODEL_FILE", None)
if model_path is None:
    # fallback: if user still has BEST_MODEL_FILE, use that
    model_path = Path(config.BEST_MODEL_FILE)

model_path = Path(model_path)

# Threshold default
best_threshold = 0.50

# Load threshold from meta json if available
meta_path = Path(str(model_path).replace(".pt", "_meta.json"))
if meta_path.exists():
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "best_threshold" in meta:
            best_threshold = float(meta["best_threshold"])
            print(f"✅ Loaded best_threshold={best_threshold:.2f} from metadata.")
    except Exception as e:
        print(f"⚠️ Could not load metadata threshold: {e}")
else:
    print("⚠️ No meta json found. Using threshold=0.50")

# Load model weights
model = LogisticRegressionTorch(input_dim).to(device)

if not model_path.exists():
    print(f"❌ WARNING: Model file not found: {model_path}")
    print("Prediction will fail until you train + save the model.")
else:
    try:
        # map_location ensures it works on CPU even if trained on GPU
        state_dict = torch.load(model_path, map_location=device)

        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading Torch model: {e}")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def build_input_from_list(values: list[float]) -> torch.Tensor:
    """
    Build model input tensor from a list of floats.
    """
    if len(values) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} features, but got {len(values)}."
        )

    arr = np.array([values], dtype=np.float32)  # shape (1, n_features)
    tensor = torch.tensor(arr, dtype=torch.float32).to(device)
    return tensor


def build_input_from_dict(payload: dict) -> torch.Tensor:
    """
    Build model input tensor from a dict:
    { "PC0": 0.12, "PC1": -1.3, ... }
    Missing columns -> error (strict)
    """
    missing = [c for c in feature_names if c not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing[:10]} ... (total missing={len(missing)})")

    row = [float(payload[c]) for c in feature_names]
    return build_input_from_list(row)


def predict_tensor(x_tensor: torch.Tensor):
    """
    Returns:
      proba (float), predicted_class (0/1 int)
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    with torch.no_grad():
        proba = float(model(x_tensor).item())
        pred = int(proba >= best_threshold)
        return proba, pred


# ------------------------------------------------------------
# Simulation Data Loading
# ------------------------------------------------------------
SIMULATION_DATA = {"normal": [], "fraud": []}

def load_simulation_data():
    """
    Load a small sample of real data for simulation.
    """
    try:
        if not config.X_RESAMPLED_FILE.exists() or not config.Y_RESAMPLED_FILE.exists():
            print("⚠️ Resampled data not found. Simulation mode disabled.")
            return

        # Load subset
        X = pd.read_csv(config.X_RESAMPLED_FILE)
        y = pd.read_csv(config.Y_RESAMPLED_FILE)
        
        # Ensure y is proper
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        
        # Combine
        df = X.copy()
        df["target"] = y.values

        # Sample 50 fraud and 50 normal
        fraud_df = df[df["target"] == 1].sample(n=min(50, len(df[df["target"]==1])), random_state=42)
        normal_df = df[df["target"] == 0].sample(n=min(50, len(df[df["target"]==0])), random_state=42)
        
        # Convert to records
        # Explicit frame cast to ensure to_dict(orient='records') is valid
        SIMULATION_DATA["fraud"] = pd.DataFrame(fraud_df.drop(columns=["target"])).to_dict(orient="records")
        SIMULATION_DATA["normal"] = pd.DataFrame(normal_df.drop(columns=["target"])).to_dict(orient="records")
        
        print(f"✅ Simulation data loaded: {len(SIMULATION_DATA['fraud'])} fraud, {len(SIMULATION_DATA['normal'])} normal.")

    except Exception as e:
        print(f"❌ Error loading simulation data: {e}")

# Load on startup
load_simulation_data()


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def home():
    """
    HTML Form page (expects index.html in templates/)
    """
    return render_template("index.html")


@app.route("/api/sample", methods=["GET"])
def get_sample():
    """
    Get a random sample for simulation.
    Query param: type=normal|fraud
    """
    sample_type = request.args.get("type", "normal")
    
    if sample_type not in SIMULATION_DATA or not SIMULATION_DATA[sample_type]:
        return jsonify({"error": "No simulation data available"}), 404
        
    # Random selection
    row = np.random.choice(SIMULATION_DATA[sample_type])
    
    return jsonify({
        "data": row,
        "type": sample_type,
        "label": "Fraud" if sample_type == "fraud" else "Normal"
    })


@app.route("/predict", methods=["POST"])
def predict_form():
    """
    Form-based prediction:
    Input: comma-separated features in form field name="data"
    """
    try:
        data_str = request.form.get("data", "")

        if not data_str.strip():
            return render_template("index.html", prediction_text="⚠️ Please enter data.")

        # Parse comma-separated features
        try:
            values = [float(x.strip()) for x in data_str.split(",")]
        except Exception:
            return render_template(
                "index.html",
                prediction_text="❌ Invalid format. Please enter comma-separated numbers.",
            )

        # Build tensor + predict
        x_tensor = build_input_from_list(values)
        proba, pred = predict_tensor(x_tensor)

        label = "Default/Fraud (1)" if pred == 1 else "Normal (0)"
        
        # Pass threshold to template
        return render_template(
            "index.html",
            prediction_text=f"Prediction: {label}",
            probability=f"{proba:.4f}",
            threshold=best_threshold
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error occurred: {str(e)}")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
    JSON API Prediction

    Accepts either:
    1) {"features":[0.1,0.2,...]}
    2) {"data":{"PC0":0.1,"PC1":0.2,...}}
    """
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        # Option 1: list
        if "features" in payload:
            values = payload["features"]
            if not isinstance(values, list):
                return jsonify({"error": "'features' must be a list"}), 400

            x_tensor = build_input_from_list([float(v) for v in values])

        # Option 2: dict
        elif "data" in payload:
            data_dict = payload["data"]
            if not isinstance(data_dict, dict):
                return jsonify({"error": "'data' must be a dict"}), 400

            x_tensor = build_input_from_dict(data_dict)

        else:
            return jsonify(
                {
                    "error": "Missing input. Provide either 'features' list or 'data' dict."
                }
            ), 400

        proba, pred = predict_tensor(x_tensor)

        return jsonify(
            {
                "prediction": pred,
                "probability": round(proba, 6),
                "threshold": best_threshold,
                "label": "Default/Fraud" if pred == 1 else "Normal",
                "n_features_expected": len(feature_names),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
