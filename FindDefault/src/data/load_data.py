# ============================================================
# Credit Card Fraud - EDA + Preprocessing (Pyrefly-safe)
# ============================================================
# This script:
# 1) Loads creditcard.csv
# 2) Runs fast EDA (missing heatmap, univariate plots, sampled pairplot, correlation heatmap)
# 3) Detects outliers (IQR method) + saves reports and plots
# 4) Preprocesses data (scales Time + Amount, saves outputs)
#
# Added:
# ✅ Outlier detection summary (IQR)
# ✅ Outlier boxplots for top outlier columns
# ✅ Amount + Time outlier plot
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# -----------------------------
# OPTIONAL GPU SUPPORT
# -----------------------------
try:
    import cupy as cp  # optional
except Exception:
    cp = None

try:
    import torch  # optional
except Exception:
    torch = None


# -----------------------------
# Add project root to sys.path
# -----------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src import config  # expects RAW_DATA_FILE, CLEANED_DATA_FILE, VISUALIZATION_DIR, PREPROCESSED_DATA_FILE, X_SCALED_FILE, Y_FILE


# -----------------------------
# Device setup (optional, Pyrefly-safe)
# -----------------------------
if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

print(f"Using device: {device}")


# -----------------------------
# Ensure directories exist
# -----------------------------
config.ensure_dirs()


# ============================================================
# BASIC DATA OVERVIEW
# ============================================================
def data_overview(df: pd.DataFrame) -> None:
    """Display dataset shape, dtypes, missing values, and stats."""
    print("\n--- Dataset Overview ---")
    print("Shape:", df.shape)

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values (per column) ---")
    print(df.isnull().sum())

    print("\n--- Statistical Summary ---")
    print(df.describe())


# ============================================================
# PLOT HELPERS
# ============================================================
def save_plot(path: Path) -> None:
    """Save current plot and close figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_missing_values(df: pd.DataFrame, save_path: Path) -> None:
    """Plot and save missing values heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    save_plot(save_path)


def univariate_continuous(df: pd.DataFrame, save_dir: Path, max_cols: int = 10) -> None:
    """
    Univariate analysis for numeric variables.
    Limits columns to prevent generating too many plots.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ["class", "y"]]
    numeric_cols = numeric_cols[:max_cols]

    for col in numeric_cols:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(x=df[col], kde=True)
        plt.title(f"Histogram of {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")

        save_plot(save_dir / f"univariate_{col}.png")


def bivariate_continuous(df: pd.DataFrame, save_dir: Path, sample_size: int = 2000) -> None:
    """
    Pairplot is heavy on large datasets. This samples safely.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ["class", "y"]]
    numeric_cols = numeric_cols[:6]

    if len(numeric_cols) < 2:
        print("Not enough numeric features for bivariate analysis.")
        return

    df_sample = df[numeric_cols].sample(n=min(sample_size, len(df)), random_state=42)

    g = sns.pairplot(df_sample, diag_kind="kde")
    g.figure.suptitle("Bivariate Analysis (Pairplot - Sampled)", y=1.02)
    g.savefig(save_dir / "bivariate_continuous_pairplot.png", dpi=300)
    plt.close("all")


def correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot correlation heatmap.
    Uses CPU pandas correlation for stability.
    """
    numeric_df = df.select_dtypes(include=["float64", "int64"]).copy()

    plt.figure(figsize=(14, 10))
    corr = numeric_df.corr()

    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        linewidths=0.3,
    )
    plt.title("Correlation Heatmap", fontsize=15)
    save_plot(save_path)


# ============================================================
# OUTLIER DETECTION (IQR METHOD)
# ============================================================
def detect_outliers_iqr(series: pd.Series, iqr_multiplier: float = 1.5) -> pd.Series:
    """
    Returns a boolean mask for outliers using the IQR method.
    Outlier if:
      x < Q1 - 1.5*IQR OR x > Q3 + 1.5*IQR
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    return (series < lower) | (series > upper)


def outlier_summary_report(df: pd.DataFrame, save_dir: Path, top_k: int = 10) -> pd.DataFrame:
    """
    Creates an outlier summary report for numeric columns and saves:
    - outlier_summary.csv
    - outlier_boxplot_top_features.png
    - outlier_amount_time.png (special focused plot)
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ["class", "y"]]

    summary_rows = []

    for col in numeric_cols:
        mask = detect_outliers_iqr(df[col])
        count_outliers = int(mask.sum())
        pct = float((count_outliers / len(df)) * 100)

        summary_rows.append(
            {
                "feature": col,
                "outlier_count": count_outliers,
                "outlier_percent": round(pct, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "outlier_percent", ascending=False
    ).reset_index(drop=True)

    # Save outlier summary
    summary_path = save_dir / "outlier_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Outlier summary saved to: {summary_path}")

    # Plot boxplots for top K outlier-heavy features
    top_features = summary_df.head(top_k)["feature"].tolist()

    if len(top_features) > 0:
        plt.figure(figsize=(14, 6))
        df[top_features].boxplot(rot=45)
        plt.title(f"Top {top_k} Features With Most Outliers (IQR)")
        save_plot(save_dir / "outlier_boxplot_top_features.png")
        print(f"✅ Outlier boxplot saved to: {save_dir / 'outlier_boxplot_top_features.png'}")

    # Special plot for Amount + Time (very important in this dataset)
    cols_special = [c for c in ["Amount", "Time"] if c in df.columns]
    if len(cols_special) > 0:
        plt.figure(figsize=(10, 5))
        df[cols_special].boxplot()
        plt.title("Outlier Boxplot: Time + Amount")
        save_plot(save_dir / "outlier_amount_time.png")
        print(f"✅ Amount/Time outlier plot saved to: {save_dir / 'outlier_amount_time.png'}")

    return summary_df


# ============================================================
# MAIN EDA FUNCTION
# ============================================================
def perform_eda(filepath: Path) -> None:
    """Run EDA and save plots + cleaned CSV."""
    print("Performing EDA on the dataset...")

    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    data_overview(df)

    plot_missing_values(df, config.VISUALIZATION_DIR / "missing_values_heatmap.png")

    print("Performing Univariate Analysis (limited)...")
    univariate_continuous(df, config.VISUALIZATION_DIR, max_cols=10)

    print("Performing Bivariate Analysis (sampled)...")
    bivariate_continuous(df, config.VISUALIZATION_DIR, sample_size=2000)

    print("Generating Correlation Heatmap...")
    correlation_heatmap(df, config.VISUALIZATION_DIR / "correlation_heatmap.png")

    # ✅ Outlier detection + report
    print("Detecting Outliers (IQR method)...")
    outlier_summary_report(df, config.VISUALIZATION_DIR, top_k=10)

    # Save cleaned copy
    df.to_csv(config.CLEANED_DATA_FILE, index=False)
    print(f"✅ Cleaned data saved to: {config.CLEANED_DATA_FILE}")


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_data(filepath: Path):
    """
    Preprocess dataset:
    - clean columns
    - split X/y
    - scale Time + Amount
    - save X_scaled, y, and full preprocessed dataset
    """
    print("\nPreprocessing dataset...")

    df = pd.read_csv(filepath)
    df.columns = df.columns.astype(str).str.strip()

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    print("\nClass distribution:")
    print(y.value_counts())

    scaler = StandardScaler()

    cols_to_scale = []
    if "Amount" in X.columns:
        cols_to_scale.append("Amount")
    if "Time" in X.columns:
        cols_to_scale.append("Time")

    if cols_to_scale:
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # Save outputs
    X.to_csv(config.X_SCALED_FILE, index=False)
    y.to_csv(config.Y_FILE, index=False)

    preprocessed_df = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1
    )
    preprocessed_df.to_csv(config.PREPROCESSED_DATA_FILE, index=False)

    print(f"\n✅ X_scaled saved to: {config.X_SCALED_FILE}")
    print(f"✅ y saved to: {config.Y_FILE}")
    print(f"✅ full preprocessed dataset saved to: {config.PREPROCESSED_DATA_FILE}")

    return X, y, preprocessed_df


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    perform_eda(config.RAW_DATA_FILE)
    X_scaled, y, preprocessed_data = preprocess_data(config.CLEANED_DATA_FILE)
