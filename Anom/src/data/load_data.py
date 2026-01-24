import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import VISUALIZATION_DIR, RAW_DATA_PATH, setup_directories

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create required folders using config.py
setup_directories()


# ============================================================
# UTILITIES
# ============================================================
def save_plot(filename: str):
    """
    Save the current matplotlib figure inside VISUALIZATION_DIR.
    Note: Folder creation is handled by config.setup_directories().
    """
    plt.tight_layout()
    plt.savefig(Path(VISUALIZATION_DIR) / filename, dpi=300)
    plt.close()


# ============================================================
# FEATURE SELECTION FOR TARGETED EDA
# ============================================================
def get_top_correlated_features(data: pd.DataFrame, target_col: str = "y", top_n: int = 10):
    """
    Identify the top N numeric features that correlate most strongly with the target.
    - Uses numeric-only columns
    - Drops duplicate label columns like y.1 (common Excel artifact)
    - Returns list of top feature names
    """
    if target_col not in data.columns:
        logging.warning(f"Target column '{target_col}' not found for correlation feature selection.")
        return [c for c in data.columns if c != target_col]

    numeric_df = data.select_dtypes(include=["number"]).copy()

    # Defensive: drop duplicate target column if present
    numeric_df = numeric_df.drop(columns=["y.1"], errors="ignore")

    if target_col not in numeric_df.columns:
        logging.warning(f"Target column '{target_col}' not numeric or missing after filtering.")
        return [c for c in numeric_df.columns if c != target_col]

    correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    correlations = correlations.drop(target_col, errors="ignore")

    top_features = correlations.head(top_n).index.tolist()
    logging.info(f"Top {top_n} Correlated Features selected for EDA: {top_features}")
    return top_features


# ============================================================
# PLOTS - BASIC EDA
# ============================================================
def plot_missing_values(data: pd.DataFrame):
    """Plot and save missing values heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    save_plot("missing_values_heatmap.png")


def plot_target_distribution(data: pd.DataFrame, target_col: str = "y"):
    """Plot and save target distribution (class balance)."""
    if target_col not in data.columns:
        logging.warning(f"Target column '{target_col}' not found - skipping target distribution plot.")
        return

    plt.figure(figsize=(6, 4))
    sns.countplot(x=data[target_col])
    plt.title("Target Distribution (Normal vs Anomaly)")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    save_plot("target_distribution.png")


def univariate_continuous(data: pd.DataFrame, features: list):
    """Univariate analysis for selected numeric features."""
    for col in features:
        if col not in data.columns:
            continue

        if not pd.api.types.is_numeric_dtype(data[col]):
            continue

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(data[col], kde=True)
        plt.title(f"Histogram of {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[col])
        plt.title(f"Boxplot of {col}")

        save_plot(f"univariate_{col}.png")


def univariate_categorical(data: pd.DataFrame):
    """Univariate analysis for categorical variables (if any exist)."""
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns

    if len(categorical_columns) == 0:
        logging.info("No categorical columns found for univariate categorical analysis.")
        return

    for col in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=data, x=col)
        plt.title(f"Bar Plot of {col}")
        plt.xticks(rotation=90)
        save_plot(f"univariate_{col}.png")


def bivariate_continuous_pairplot(data: pd.DataFrame, features: list, target_col: str = "y", max_features: int = 5):
    """
    Pairplot using top features (limited for performance).
    """
    if target_col not in data.columns:
        logging.warning(f"Target column '{target_col}' not found - skipping pairplot.")
        return

    selected_features = [f for f in features if f in data.columns][:max_features]
    if len(selected_features) < 2:
        logging.warning("Not enough features to generate pairplot - skipping.")
        return

    cols_to_plot = selected_features + [target_col]
    df_sample = data[cols_to_plot].sample(min(len(data), 500), random_state=42)

    g = sns.pairplot(df_sample, hue=target_col, diag_kind="kde")
    g.fig.suptitle("Bivariate Analysis (Top Features)", y=1.02)
    g.savefig(Path(VISUALIZATION_DIR) / "bivariate_continuous_pairplot.png", dpi=300)
    plt.close("all")


def correlation_heatmap(data: pd.DataFrame):
    """Plot correlation heatmap for numeric features."""
    numeric_df = data.select_dtypes(include=["number"]).drop(columns=["y.1"], errors="ignore")

    plt.figure(figsize=(12, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")


# ============================================================
# PLOTS - ADVANCED EDA
# ============================================================
def plot_class_distribution_by_feature(data: pd.DataFrame, features: list, target_col: str = "y"):
    """
    Plot KDE distributions separated by class for selected numeric features.
    """
    if target_col not in data.columns:
        logging.warning(f"Target column '{target_col}' not found - skipping class distribution plots.")
        return

    for col in features:
        if col not in data.columns:
            continue

        if not pd.api.types.is_numeric_dtype(data[col]):
            continue

        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=data, x=col, hue=target_col, fill=True, common_norm=False)
        plt.title(f"Distribution of {col} by Class (Normal vs Anomaly)")
        save_plot(f"advanced_kde_{col}.png")


def plot_pca_2d(data: pd.DataFrame, target_col: str = "y"):
    """
    PCA projection down to 2D using numeric-only features.
    Safe against 'time' columns and any non-numeric leakage.
    """
    logging.info("Generating PCA 2D Projection...")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if target_col not in data.columns:
        logging.warning(f"Target column '{target_col}' not found - skipping PCA.")
        return

    numeric_df = data.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.drop(columns=["y.1"], errors="ignore")

    if target_col not in numeric_df.columns:
        logging.warning(f"Target '{target_col}' not present in numeric data - skipping PCA.")
        return

    X = numeric_df.drop(columns=[target_col], errors="ignore")
    y = numeric_df[target_col]

    # Replace infinities and drop NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if X.shape[1] < 2:
        logging.warning("Not enough numeric features for PCA - skipping.")
        return

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df[target_col] = y.values

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=target_col, alpha=0.6)

    explained_var = pca.explained_variance_ratio_.sum()
    plt.title(f"PCA 2D Projection (Explained Var: {explained_var:.2%})")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})")

    save_plot("advanced_pca_projection.png")


# ============================================================
# EDA ORCHESTRATORS
# ============================================================
def perform_initial_eda(data: pd.DataFrame):
    """
    Perform targeted EDA.
    Produces:
    - Missing values heatmap
    - Target distribution plot
    - Univariate plots (top correlated features)
    - Pairplot (top 5)
    - Correlation heatmap
    """
    logging.info("Starting Targeted EDA...")

    top_features = get_top_correlated_features(data, target_col="y", top_n=10)

    plot_missing_values(data)
    plot_target_distribution(data, target_col="y")

    logging.info("Performing Focused Univariate Analysis...")
    univariate_continuous(data, top_features)
    univariate_categorical(data)

    logging.info("Performing Focused Bivariate Analysis...")
    bivariate_continuous_pairplot(data, top_features, target_col="y", max_features=5)

    logging.info("Generating Correlation Heatmap...")
    correlation_heatmap(data)

    logging.info(f"EDA plots saved to: {VISUALIZATION_DIR}")
    return top_features


def perform_advanced_eda(data: pd.DataFrame, top_features: list):
    """
    Advanced EDA:
    - KDE by class
    - PCA projection
    """
    logging.info("Starting ADVANCED EDA...")

    plot_class_distribution_by_feature(data, top_features, target_col="y")

    try:
        plot_pca_2d(data, target_col="y")
    except Exception as e:
        logging.warning(f"Could not perform PCA: {e}")

    logging.info("Advanced EDA completed.")


# ============================================================
# LOAD + CLEAN DATA
# ============================================================
def load_and_clean_data(filepath: Path):
    """
    Load data from Excel and perform basic cleaning.
    Cleaning includes:
    - Drop duplicate label column y.1 if present
    - Drop datetime columns (like time)
    - Ensure y exists and is int
    Returns:
    - cleaned full dataframe (including y)
    """
    logging.info(f"Loading data from: {filepath}")

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    data = pd.read_excel(filepath)
    
    # Clean column names (remove hidden newlines/spaces)
    data.columns = data.columns.astype(str).str.strip()

    # ============================================================
    # BASIC PROFILING
    # ============================================================
    logging.info("\n" + "=" * 70)
    logging.info("INITIAL DATA ANALYSIS (Before Cleaning)")
    logging.info("=" * 70)

    logging.info(f"Dataset Shape: {data.shape}")
    logging.info(f"Data Types:\n{data.dtypes}")
    logging.info(f"Null Values Count:\n{data.isnull().sum()}")

    if "y" in data.columns:
        logging.info(f"Target Distribution:\n{data['y'].value_counts()}")
    else:
        logging.warning("Target column 'y' not found in raw data!")

    # ============================================================
    # CLEANING
    # ============================================================
    if "y.1" in data.columns:
        logging.info("⚠️ Dropping duplicate label column: y.1")
        data = data.drop(columns=["y.1"])

    # Drop datetime columns safely (catches datetime64[ns])
    datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
    if len(datetime_cols) > 0:
        logging.info(f"⚠️ Dropping datetime columns: {list(datetime_cols)}")
        data = data.drop(columns=datetime_cols)

    if "y" not in data.columns:
        raise ValueError("Target column 'y' not found after cleaning.")

    data["y"] = data["y"].astype(int)

    logging.info("\n" + "=" * 70)
    logging.info("DATA AFTER CLEANING")
    logging.info("=" * 70)
    logging.info(f"Cleaned Dataset Shape: {data.shape}")
    logging.info(f"Cleaned Data Types:\n{data.dtypes}")

    return data
