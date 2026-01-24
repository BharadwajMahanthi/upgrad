"""
Fire & Smoke Detection Web Application
======================================

This Flask-based web application provides real-time fire and smoke detection capabilities
using a deep learning Classification model (MobileNetV2). It includes advanced features
for image enhancement, Grad-CAM localization, and live video streaming.

Key Features
------------
1. **Real-time MobileNetV2 Inference**:
   - Uses a pre-trained MobileNetV2 model fine-tuned for Fire, Smoke, and Neutral classes.
   - Includes temporal smoothing to reduce flicker in live video feeds.

2. **Grad-CAM Localization**:
   - Even though the model is a classifier, this app uses Gradient-weighted Class Activation Mapping (Grad-CAM)
   - to visualize *where* the model is looking.
   - Generates bounding boxes around Fire/Smoke regions by thresholding the activation heatmap.

3. **Advanced Image Enhancement**:
   - `suppress_reflections`: Removes glare/specular highlights common in glass reflection scenarios.
   - `apply_clahe`: Contrast Limited Adaptive Histogram Equalization for better visibility in smoke/fog.
   - `gamma_correction`: Adapts to harsh or dim lighting conditions.
   - `denoise_image` & `unsharp_mask`: Reduces sensor noise and sharpens edges for clearer features.

4. **Live Streaming**:
   - Processes base64-encoded frames from the frontend via `/predict_live`.
   - Optimized for low latency using threaded requests and efficient resizing.

Endpoints
---------
- `GET /`: Renders the main web interface (`index.html`).
- `POST /predict`: Handles single image uploads for detection. Returns class, confidence, and bounding box.
- `POST /predict_live`: Handles real-time video frames (base64). Returns prediction and bounding box JSON.

Usage
-----
Run the application with:
    python app.py

Access the interface at http://localhost:5000
"""
import os
import base64
import io
import threading
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Centralized Config
import config

# ============================================================
# FLASK APP INIT
# ============================================================
app = Flask(__name__)

# ============================================================
# MODEL LOAD
# ============================================================
MODEL_PATH = os.path.join(config.OUTPUT_DIR, config.MODEL_FILENAME)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fire_detection.h5")

model = None
grad_model = None
LAST_CONV_LAYER_PATH = None

# ✅ Lock to prevent Grad-CAM race conditions under threaded Flask
cam_lock = threading.Lock()

# ============================================================
# QUALITY / ROBUSTNESS SETTINGS (you can tune these)
# ============================================================
ENABLE_QUALITY_ENHANCEMENT = True  # turn ON/OFF enhancements
ENABLE_REFLECTION_SUPPRESSION = True  # inpaint glare reflections
ENABLE_CLAHE = True  # contrast boost
ENABLE_DENOISE = True  # reduce sensor noise
ENABLE_SHARPEN = True  # enhance edges
ENABLE_GAMMA = True  # adapt brightness

# For live webcam: higher gives smoother predictions (less flicker)
ENABLE_TEMPORAL_SMOOTHING = True
SMOOTHING_ALPHA = 0.6  # 0.0 = no smoothing, 0.9 = very stable but slower reaction

# Keep last prediction probs for smoothing
_last_probs = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Model loaded successfully from: {MODEL_PATH}")

        # ✅ Warm-up model once (important for Keras 3)
        try:
            dummy = np.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1], 3), dtype=np.float32)
            _ = model(dummy, training=False)
            print("[INFO] Model warm-up completed ✅")
        except Exception as build_err:
            print(f"[WARNING] Model warm-up failed: {build_err}")
    else:
        print(f"[WARNING] Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")


# ============================================================
# IMAGE ENHANCEMENT (GLASS REFLECTION + QUALITY BOOST)
# ============================================================
def gamma_correction(bgr, gamma=1.1):
    """
    Gamma correction: gamma > 1 darkens slightly, gamma < 1 brightens.
    Helpful for harsh lighting and washed-out frames.
    """
    if gamma <= 0:
        return bgr
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(bgr, table)


def suppress_reflections(bgr):
    """
    Reduce strong glass reflections/specular highlights.
    Strategy:
      - Find pixels with very HIGH brightness (V) but LOW saturation (S)
      - Inpaint those hotspots
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Reflection-like areas: very bright but low saturation (white glare)
    glare_mask = cv2.inRange(hsv, (0, 0, 220), (180, 70, 255))

    # Expand mask slightly to cover glare edges
    kernel = np.ones((5, 5), np.uint8)
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

    # Inpaint glare regions
    inpainted = cv2.inpaint(bgr, glare_mask, 5, cv2.INPAINT_TELEA)
    return inpainted


def apply_clahe(bgr):
    """
    Improve contrast safely using CLAHE on L channel (LAB color space).
    Very useful for smoke visibility and foggy reflections.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise_image(bgr):
    """
    Bilateral filter keeps edges while reducing noise.
    """
    return cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)


def unsharp_mask(bgr, amount=1.2):
    """
    Sharpen image using unsharp masking.
    """
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.2, sigmaY=1.2)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return sharp


def high_quality_resize(bgr, target_size):
    """
    Better resizing for small webcam frames.
    """
    target_w, target_h = target_size[0], target_size[1]
    h, w = bgr.shape[:2]

    # If upscale -> cubic/bicubic looks better
    if target_w > w or target_h > h:
        interp = cv2.INTER_CUBIC
    else:
        # If downscale -> INTER_AREA is best
        interp = cv2.INTER_AREA

    return cv2.resize(bgr, (target_w, target_h), interpolation=interp)


def enhance_frame_for_model(rgb_np):
    """
    Input: RGB numpy image (H,W,3) uint8
    Output: enhanced RGB numpy image (IMG_SIZE)
    """
    # Convert RGB -> BGR for OpenCV pipeline
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    if ENABLE_QUALITY_ENHANCEMENT:
        if ENABLE_REFLECTION_SUPPRESSION:
            bgr = suppress_reflections(bgr)

        if ENABLE_DENOISE:
            bgr = denoise_image(bgr)

        if ENABLE_CLAHE:
            bgr = apply_clahe(bgr)

        if ENABLE_GAMMA:
            # small gamma helps reflections / overexposure
            bgr = gamma_correction(bgr, gamma=1.15)

        if ENABLE_SHARPEN:
            bgr = unsharp_mask(bgr, amount=1.1)

    # Resize at the end with high-quality interpolation
    bgr = high_quality_resize(bgr, (config.IMG_SIZE[0], config.IMG_SIZE[1]))

    # Convert back BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ============================================================
# GRAD-CAM HELPERS
# ============================================================
def find_last_conv_layer_path(m: tf.keras.Model):
    """
    Finds the best last 4D convolution-like layer INSIDE MobileNetV2 backbone.
    Returns a path list like: [base_model_name, last_4d_layer_name]
    """
    if not hasattr(m, "layers"):
        return None

    # Find backbone model inside Sequential
    backbone = None
    for lyr in m.layers:
        if isinstance(lyr, tf.keras.Model):
            backbone = lyr
            break

    if backbone is None:
        return None

    # Find last 4D layer INSIDE backbone (not the backbone itself)
    last_4d = None
    for lyr in backbone.layers[::-1]:
        try:
            shape = lyr.output_shape
            if isinstance(shape, list):
                shape = shape[0]
            if shape is not None and len(shape) == 4:
                last_4d = lyr
                break
        except Exception:
            continue

    if last_4d is None:
        return None

    return [backbone.name, last_4d.name]


def get_layer_by_path(m: tf.keras.Model, path):
    curr = m
    for name in path:
        curr = curr.get_layer(name)
    return curr


def build_grad_model_once():
    global grad_model, LAST_CONV_LAYER_PATH

    if model is None:
        return

    LAST_CONV_LAYER_PATH = find_last_conv_layer_path(model)
    if not LAST_CONV_LAYER_PATH:
        print("[WARNING] Could not find a valid last conv layer for Grad-CAM.")
        return

    try:
        target_layer = get_layer_by_path(model, LAST_CONV_LAYER_PATH)

        grad_model_local = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.outputs[0]],
        )

        # Warm-up grad model
        dummy = np.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1], 3), dtype=np.float32)
        _ = grad_model_local(dummy, training=False)

        grad_model = grad_model_local
        print(f"[INFO] Using Grad-CAM layer path: {LAST_CONV_LAYER_PATH}")
        print("[INFO] Grad-CAM model built & warmed ✅")

    except Exception as e:
        grad_model = None
        print(f"[WARNING] Failed to build Grad-CAM model: {e}")


def get_grad_cam_heatmap(img_array):
    if grad_model is None:
        raise RuntimeError("Grad-CAM model not initialized.")

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array, training=False)
        top_pred_index = tf.argmax(preds[0])
        class_score = preds[:, top_pred_index]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise ValueError("Gradients are None. Cannot compute Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]  # (H, W, C)

    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def find_best_box(heatmap, threshold=0.4):
    heatmap = cv2.resize(heatmap, (config.IMG_SIZE[1], config.IMG_SIZE[0]))
    mask = (heatmap > threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return {
        "x": float(x) / config.IMG_SIZE[1],
        "y": float(y) / config.IMG_SIZE[0],
        "w": float(w) / config.IMG_SIZE[1],
        "h": float(h) / config.IMG_SIZE[0],
    }


# ============================================================
# PREPROCESS + PREDICT
# ============================================================
def prepare_image_pil(img_pil: Image.Image):
    """
    Convert + fix EXIF orientation + enhance + preprocess for MobileNetV2.
    """
    # Fix rotated phone images automatically
    img_pil = ImageOps.exif_transpose(img_pil)

    # Convert to RGB
    img_rgb = img_pil.convert("RGB")

    # Convert to numpy RGB
    rgb_np = np.array(img_rgb).astype(np.uint8)

    # ✅ Enhance quality + reflections suppression
    rgb_np = enhance_frame_for_model(rgb_np)

    # Convert to float32 batch
    img_array = rgb_np.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Correct preprocessing for MobileNetV2
    img_array = preprocess_input(img_array)

    return img_array


def run_prediction(img_array, live_mode=False):
    """
    Returns (predicted_class, confidence_float, probs_array)
    Includes optional temporal smoothing for live feed.
    """
    global _last_probs

    preds = model.predict(img_array, verbose=0)[0]  # shape (num_classes,)
    preds = preds.astype(np.float32)

    # ✅ Temporal smoothing (reduces flicker & reflection spikes)
    if live_mode and ENABLE_TEMPORAL_SMOOTHING:
        if _last_probs is None:
            _last_probs = preds
        else:
            _last_probs = (SMOOTHING_ALPHA * _last_probs) + ((1.0 - SMOOTHING_ALPHA) * preds)
        preds_smooth = _last_probs
    else:
        preds_smooth = preds

    idx = int(np.argmax(preds_smooth))
    predicted_class = config.CLASSES[idx]
    confidence = float(preds_smooth[idx])

    return predicted_class, confidence, preds_smooth


# ✅ Build grad_model AFTER model is loaded
if model is not None:
    build_grad_model_once()


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server!"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided!"}), 400

    try:
        file = request.files["file"]
        img_pil = Image.open(file.stream)

        img_array = prepare_image_pil(img_pil)
        predicted_class, confidence, _ = run_prediction(img_array, live_mode=False)

        box = None
        if predicted_class in ["fire", "Smoke"] and confidence > 0.4 and grad_model is not None:
            try:
                with cam_lock:
                    heatmap = get_grad_cam_heatmap(img_array)
                box = find_best_box(heatmap, threshold=0.4)
            except Exception as cam_err:
                print(f"[DEBUG] CAM Error: {cam_err}")

        return jsonify(
            {
                "predicted_class": predicted_class,
                "confidence": f"{confidence * 100:.2f}%",
                "box": box,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_live", methods=["POST"])
def predict_live():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Empty frame"}), 400

    try:
        header, encoded = data["image"].split(",", 1)
        image_data = base64.b64decode(encoded)

        img_pil = Image.open(io.BytesIO(image_data))

        img_array = prepare_image_pil(img_pil)
        predicted_class, confidence, _ = run_prediction(img_array, live_mode=True)

        box = None
        if predicted_class in ["fire", "Smoke"] and confidence > 0.4 and grad_model is not None:
            try:
                with cam_lock:
                    heatmap = get_grad_cam_heatmap(img_array)
                box = find_best_box(heatmap, threshold=0.4)
            except Exception as cam_err:
                print(f"[DEBUG] CAM Error: {cam_err}")

        return jsonify(
            {
                "class": predicted_class,
                "confidence": confidence,
                "box": box,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
