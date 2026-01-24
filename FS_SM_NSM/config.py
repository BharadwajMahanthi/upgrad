import os

# ============================================================
# PROJECT PATHS
# ============================================================
# Absolute path to the directory where config.py is located (project root)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# ============================================================
# MODEL SETTINGS
# ============================================================
# ✅ MUST match folder names EXACTLY (case + spacing)
CLASSES = ["non fire", "fire", "Smoke"]

# ✅ 128 is good, but for better accuracy behind glass try 160 or 192
# (Keep 128 if training is too slow on CPU)
IMG_SIZE = (160, 160)

# ✅ CPU friendly = 16 or 32 (try 16 if RAM/CPU struggles)
# ✅ If you have GPU, set 64
BATCH_SIZE = 16

MODEL_FILENAME = "fire_detection_best_model.keras"
CLASS_MAP_FILENAME = "classes.json"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
# ✅ Slightly higher validation split gives more reliable estimate
VAL_SPLIT = 0.15
SEED = 42

# ✅ KerasTuner - keep it small on CPU
MAX_TUNER_TRIALS = 4
TUNER_EPOCHS = 8

# ✅ Two-phase training will happen in your updated training code:
# Phase-1 uses TRAIN_EPOCHS
# Phase-2 will auto run with max(5, TRAIN_EPOCHS//2)
TRAIN_EPOCHS = 18

# ============================================================
# SAFETY / DEPLOYMENT THRESHOLDS (OPTIONAL but recommended)
# ============================================================
# Used during inference to avoid false alarms from reflections
FIRE_SMOKE_CONFIDENCE_THRESHOLD = 0.55

# For live mode: require confirmation across frames
LIVE_CONFIRMATION_FRAMES = 3
