import pandas as pd
import numpy as np
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, KBinsDiscretizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# OPTIONAL GPU (not required for sklearn/SMOTE)
try:
    import cupy as cp  # optional
except Exception:
    cp = None

try:
    import torch
except Exception:
    torch = None


# ------------------------------------------------------------
# Add project root to sys.path
# ------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src import config


# ------------------------------------------------------------
# Device setup (Pyrefly-safe)
# ------------------------------------------------------------
if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

print(f"Using device: {device}")


# ============================================================
# LOAD DATA
# ============================================================
def load_preprocessed_data():
    """
    Load preprocessed data from disk.
    Expects:
    - X_scaled.csv
    - y.csv
    """
    X_scaled = pd.read_csv(config.X_SCALED_FILE)
    y_df = pd.read_csv(config.Y_FILE)

    # Ensure column names are clean
    X_scaled.columns = X_scaled.columns.astype(str).str.strip()

    # Extract y as 1D numpy array
    if isinstance(y_df, pd.DataFrame):
        if y_df.shape[1] == 1:
            y = y_df.iloc[:, 0]
        else:
            raise ValueError("y file contains multiple columns. Expected single target column.")
    else:
        y = y_df

    # ✅ Pyrefly-safe: force numpy ndarray[int]
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    return X_scaled, y


# ============================================================
# PLOTS
# ============================================================
def plot_class_distribution(y: np.ndarray, title: str, save_path: Path):
    """Plot and save class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# CLEANING
# ============================================================
def drop_invalid_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns containing NaN / inf values after replacement.
    """
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    invalid_cols = X.columns[X.isna().any()].tolist()
    if invalid_cols:
        print(f"⚠️ Dropping columns with NaN/Infinity values: {invalid_cols}")
        X = X.drop(columns=invalid_cols)

    return X


# ============================================================
# FEATURE ENGINEERING + SMOTE
# ============================================================
def build_features(X: pd.DataFrame, y: np.ndarray):
    """
    Feature Engineering Pipeline:
    1) Drop invalid columns
    2) PolynomialFeatures degree=2
    3) Binning (Amount)
    4) Log Transform (Amount)
    5) PCA retain 95% variance
    6) SMOTE
    """
    config.ensure_dirs()

    # ✅ Ensure y is proper numpy array (Pyrefly-safe)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    # ----------------------------
    # Clean invalid cols
    # ----------------------------
    X_clean = drop_invalid_columns(X)

    # Plot distribution BEFORE SMOTE
    plot_class_distribution(
        y,
        "Class Distribution Before SMOTE",
        config.VISUALIZATION_DIR / "class_distribution_before_smote.png",
    )

    # ----------------------------
    # 1) Polynomial Features
    # ----------------------------
    print("Generating Polynomial Features (degree=2)...")

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_np = poly.fit_transform(X_clean.values)

    poly_feature_names = poly.get_feature_names_out(X_clean.columns)
    X_poly = pd.DataFrame(X_poly_np, columns=poly_feature_names)

    # ----------------------------
    # 2) Binning (Amount)
    # ----------------------------
    if "Amount" in X_clean.columns and "Amount" in X_poly.columns:
        print("Adding Amount binning feature...")

        binning = KBinsDiscretizer(
            n_bins=10,
            encode="ordinal",
            strategy="uniform",
        )

        # ✅ Pyrefly-safe: force numpy and flatten (avoid .ravel() on list-like)
        amount_binned_raw = binning.fit_transform(X_poly[["Amount"]])
        amount_binned = np.asarray(amount_binned_raw, dtype=np.float64).reshape(-1)

        X_poly["Amount_binned"] = amount_binned
    else:
        print("⚠️ 'Amount' column not found. Skipping binning feature.")

    # ----------------------------
    # 3) Log Transform (Amount)
    # ----------------------------
    if "Amount" in X_poly.columns:
        print("Adding log(Amount) feature...")

        log_transformer = FunctionTransformer(np.log1p, validate=True)

        safe_amount = np.clip(X_poly[["Amount"]].values, 0, None)
        log_amount = np.asarray(log_transformer.transform(safe_amount), dtype=np.float64).reshape(-1)

        X_poly["log_Amount"] = log_amount
    else:
        print("⚠️ 'Amount' column not found. Skipping log transform.")

    # ----------------------------
    # 4) PCA (retain 95% variance)
    # ----------------------------
    print("Applying PCA (retain 95% variance)...")

    pca = PCA(n_components=0.95, random_state=42)
    X_pca_np = pca.fit_transform(X_poly.values)

    X_reduced = pd.DataFrame(
        X_pca_np,
        columns=[f"PC{i}" for i in range(X_pca_np.shape[1])],
    )

    print(f"PCA Reduced shape: {X_reduced.shape} (components retained = {X_reduced.shape[1]})")

    # ----------------------------
    # 5) SMOTE
    # ----------------------------
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)

    # ✅ Pyrefly-safe: y is guaranteed numpy int array
    X_resampled, y_resampled = smote.fit_resample(X_reduced, y)

    # Plot distribution AFTER SMOTE
    plot_class_distribution(
        np.asarray(y_resampled, dtype=np.int64),
        "Class Distribution After SMOTE",
        config.VISUALIZATION_DIR / "class_distribution_after_smote.png",
    )

    print(f"Original dataset shape: X={X_clean.shape}, y={y.shape}")
    print(f"Resampled dataset shape: X={X_resampled.shape}, y={np.asarray(y_resampled).shape}")

    # ----------------------------
    # Save outputs
    # ----------------------------
    X_resampled.to_csv(config.X_RESAMPLED_FILE, index=False)
    pd.DataFrame(np.asarray(y_resampled, dtype=np.int64), columns=["Class"]).to_csv(
        config.Y_RESAMPLED_FILE, index=False
    )

    print(f"✅ Feature-engineered data saved to: {config.X_RESAMPLED_FILE}")
    print(f"✅ Resampled target saved to: {config.Y_RESAMPLED_FILE}")

    return X_resampled, np.asarray(y_resampled, dtype=np.int64)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    X_scaled, y = load_preprocessed_data()
    X_resampled, y_resampled = build_features(X_scaled, y)
