import os
from pathlib import Path

# -----------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------

# Project Root: /Anom/
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data Directories
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'AnomaData.xlsx'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model Directories
MODELS_DIR = ROOT_DIR / 'models'
MODEL_SAVE_PATH = MODELS_DIR / 'best_model.pkl'
MODEL_RESULTS_PATH = MODELS_DIR / 'model_results.txt'

# Report & Visualization Directories
REPORTS_DIR = ROOT_DIR / 'reports'
VISUALIZATION_DIR = ROOT_DIR / 'src' / 'visualization'

# Set Matplotlib Backend to Agg (Non-interactive) to prevent main thread errors
import matplotlib
matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.3

# Ensure directories exist
def setup_directories():
    directories = [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, VISUALIZATION_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    setup_directories()
    print(f"Project configuration loaded. Root: {ROOT_DIR}")
