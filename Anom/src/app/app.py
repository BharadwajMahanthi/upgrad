import logging
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import sys
import os

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import MODEL_SAVE_PATH
from src.features.build_features import CustomFeatureEngineer # CRITICAL for unpickling
from sklearn import set_config

# Force sklearn to output pandas DataFrames by default
# This ensures pipelines (like SimpleImputer) return DataFrames so CustomFeatureEngineer works
set_config(transform_output="pandas")


# ============================================================
# FLASK APP INIT
# ============================================================
app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Flask Application Starting...")


# ============================================================
# LOAD MODEL (supports both "model only" and "bundle dict")
# ============================================================
logging.info(f"Loading model from {MODEL_SAVE_PATH}...")
try:
    loaded_obj = joblib.load(MODEL_SAVE_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model from {MODEL_SAVE_PATH}: {e}")
    raise e

# New format (recommended): {"model": model, "threshold": float, "feature_names": list[str]}
if isinstance(loaded_obj, dict) and "model" in loaded_obj:
    model = loaded_obj["model"]
    threshold = float(loaded_obj.get("threshold", 0.5))
    feature_names = loaded_obj.get("feature_names", None)
    logging.info(f"Loaded Bundle: Threshold={threshold}, Features Known={'Yes' if feature_names else 'No'}")
else:
    # Old format: directly saved sklearn model
    model = loaded_obj
    threshold = 0.5
    feature_names = None
    logging.info("Loaded legacy model format (no threshold/features in bundle). Using defaults.")


def _infer_feature_names(model_obj):
    """
    Infer expected feature names (best effort).
    Priority:
    1) If model bundle included feature_names -> use them
    2) If sklearn model exposes feature_names_in_ -> use them
    3) Otherwise return None and fallback to numeric array
    """
    if feature_names is not None:
        # Clean possible whitespace/newlines
        return [str(c).strip() for c in feature_names]

    if hasattr(model_obj, "feature_names_in_"):
        return [str(c).strip() for c in list(model_obj.feature_names_in_)]

    # Pipeline support: sometimes final estimator has feature_names_in_
    if hasattr(model_obj, "named_steps"):
        for step_name, step in model_obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return [str(c).strip() for c in list(step.feature_names_in_)]

    return None


EXPECTED_FEATURES = _infer_feature_names(model)
EXPECTED_N = len(EXPECTED_FEATURES) if EXPECTED_FEATURES is not None else None


# ============================================================
# INPUT PARSERS
# ============================================================
def parse_comma_string_to_dataframe(data_str: str) -> pd.DataFrame:
    """
    Convert comma-separated string input into DataFrame using expected feature names if available.
    """
    # Convert to float list
    values = [float(x.strip()) for x in data_str.split(",") if x.strip() != ""]

    # Validate length if we know expected size
    if EXPECTED_N is not None and len(values) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} features, but got {len(values)}.")

    # Build dataframe
    if EXPECTED_FEATURES is not None:
        df = pd.DataFrame([values], columns=EXPECTED_FEATURES)
    else:
        df = pd.DataFrame([values])

    return df


def parse_json_to_dataframe(payload) -> pd.DataFrame:
    """
    Accept JSON payload in two formats:
    1) dict of feature_name -> value
    2) list of values [v1, v2, ...]
    """
    if isinstance(payload, dict):
        # Dictionary of feature:value
        df = pd.DataFrame([payload])
        df.columns = df.columns.astype(str).str.strip()

        if EXPECTED_FEATURES is not None:
            # Ensure exactly the expected columns exist
            missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
            extra = [c for c in df.columns if c not in EXPECTED_FEATURES]

            if missing:
                raise ValueError(f"Missing features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if extra:
                # Not fatal, but safer to drop unexpected columns
                df = df[EXPECTED_FEATURES]

            df = df[EXPECTED_FEATURES]

        return df

    if isinstance(payload, list):
        values = [float(v) for v in payload]

        if EXPECTED_N is not None and len(values) != EXPECTED_N:
            raise ValueError(f"Expected {EXPECTED_N} features, but got {len(values)}.")

        if EXPECTED_FEATURES is not None:
            return pd.DataFrame([values], columns=EXPECTED_FEATURES)
        return pd.DataFrame([values])

    raise ValueError("Invalid JSON format. Use a dict {feature:value} or a list [v1, v2, ...].")


def predict_with_threshold(model_obj, X_df: pd.DataFrame, threshold_value: float):
    """
    Predict using probability threshold if predict_proba exists.
    Otherwise fallback to model.predict().
    """
    if hasattr(model_obj, "predict_proba"):
        prob = model_obj.predict_proba(X_df)[:, 1]
        pred = (prob >= threshold_value).astype(int)
        return int(pred[0]), float(prob[0])

    # fallback
    pred = model_obj.predict(X_df)
    return int(pred[0]), None


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def home():
    """
    Home page with input form.
    """
    logging.info("Home page accessed.")
    """
    Home page with input form.
    """
    return render_template(
        "index.html",
        prediction_text="",
        expected_features=EXPECTED_N,
        threshold=threshold
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    HTML Form Prediction Endpoint
    Expects comma-separated values in form field "data"
    """
    try:
        data_str = request.form.get("data", "")
        logging.info(f"Received Form Input: {data_str}")

        if not data_str.strip():
            logging.warning("Empty input received.")
            return render_template(
                "index.html",
                prediction_text="❌ Please enter valid comma-separated numeric input.",
                expected_features=EXPECTED_N,
                threshold=threshold
            )

        X_df = parse_comma_string_to_dataframe(data_str)

        pred, prob = predict_with_threshold(model, X_df, threshold)
        
        logging.info(f"Prediction Result: Class={pred}, Prob={prob}, Threshold={threshold}")

        label = "Anomaly 🚨" if pred == 1 else "Normal ✅"
        if prob is not None:
            msg = f"Prediction: {pred} ({label}) | Probability={prob:.4f} | Threshold={threshold:.2f}"
        else:
            msg = f"Prediction: {pred} ({label})"

        return render_template(
            "index.html",
            prediction_text=msg,
            expected_features=EXPECTED_N,
            threshold=threshold
        )

    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return render_template(
            "index.html",
            prediction_text=f"❌ Error occurred: {str(e)}",
            expected_features=EXPECTED_N,
            threshold=threshold
        )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
    JSON API Endpoint
    Accepts:
    - {"data": [v1, v2, ...]}
    OR
    - {"data": {"x1": v1, "x2": v2, ...}}
    Returns JSON with prediction + probability (if available)
    """
    try:
        payload = request.get_json(silent=True)
        logging.info(f"Received API Payload: {payload}")

        if payload is None or "data" not in payload:
            logging.warning("Missing 'data' in API payload.")
            return jsonify(
                {
                    "error": "Missing JSON body. Provide {'data': [...]} or {'data': {...}}"
                }
            ), 400

        X_df = parse_json_to_dataframe(payload["data"])

        pred, prob = predict_with_threshold(model, X_df, threshold)
        logging.info(f"API Prediction: Class={pred}, Prob={prob}")

        response = {
            "prediction": pred,
            "label": "anomaly" if pred == 1 else "normal",
            "threshold": threshold,
        }

        if prob is not None:
            response["probability"] = prob

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
