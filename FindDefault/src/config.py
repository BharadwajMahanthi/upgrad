import os
from pathlib import Path
from typing import List


# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DIRECTORIES
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALIZATION_DIR = PROJECT_ROOT / "src" / "visualization"


# ============================================================
# FILE PATHS
# ============================================================
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"

CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
X_SCALED_FILE = PROCESSED_DATA_DIR / "X_scaled.csv"
Y_FILE = PROCESSED_DATA_DIR / "y.csv"

X_RESAMPLED_FILE = PROCESSED_DATA_DIR / "X_features_resampled.csv"
Y_RESAMPLED_FILE = PROCESSED_DATA_DIR / "y_resampled.csv"

PREPROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "preprocessed_data.csv"

# Model outputs
BEST_MODEL_FILE = MODELS_DIR / "best_lgr_model.pkl"         # sklearn models
BEST_TORCH_MODEL_FILE = MODELS_DIR / "best_logreg_torch.pt" # torch models
MODEL_RESULTS_FILE = MODELS_DIR / "logreg_model_results.pdf"


# ============================================================
# GLOBAL CONSTANTS
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
TARGET_COL = "Class"


# ============================================================
# DIRECTORY SETUP
# ============================================================
def ensure_dirs() -> None:
    """Ensure all critical directories exist."""
    for directory in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        VISUALIZATION_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)


# ============================================================
# FEATURE NAME INFERENCE (NO HARDCODING ✅)
# ============================================================
def get_feature_names(prefer_resampled: bool = True) -> List[str]:
    """
    Automatically detect feature names from saved datasets.
    Priority:
    1) X_RESAMPLED_FILE (feature engineered final set)
    2) X_SCALED_FILE (scaled raw features)
    3) RAW_DATA_FILE minus TARGET_COL
    """
    # 1) Resampled engineered features
    if prefer_resampled and X_RESAMPLED_FILE.exists():
        df = _safe_read_csv_head(X_RESAMPLED_FILE)
        return [str(c).strip() for c in df.columns]

    # 2) Scaled features
    if X_SCALED_FILE.exists():
        df = _safe_read_csv_head(X_SCALED_FILE)
        return [str(c).strip() for c in df.columns]

    # 3) Raw file fallback
    if RAW_DATA_FILE.exists():
        df = _safe_read_csv_head(RAW_DATA_FILE)
        cols = [str(c).strip() for c in df.columns]
        return [c for c in cols if c != TARGET_COL]

    return []


def _safe_read_csv_head(path: Path):
    """
    Read only the header + first row (fast and safe).
    """
    import pandas as pd

    df = pd.read_csv(path, nrows=1)
    df.columns = df.columns.astype(str).str.strip()
    return df


if __name__ == "__main__":
    ensure_dirs()
    print(f"✅ Config loaded | Project root: {PROJECT_ROOT}")
    print(f"✅ Feature names detected: {len(get_feature_names())}")
