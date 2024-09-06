import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, KBinsDiscretizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import cupy as cp  # cuPy for GPU-accelerated array processing
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_preprocessed_data():
    """Load preprocessed data from the processed folder."""
    X_scaled = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\processed\X_scaled.csv')
    y = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\processed\y.csv')
    
    # Ensure 'y' is a 1D array (flatten it)
    y = y.values.ravel()
    
    return X_scaled, y

def plot_class_distribution(y, title, save_path):
    """Plot class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def drop_invalid_columns(X):
    """Drop columns that contain NaN, infinite, or very large/small values."""
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop columns that contain any NaN values
    invalid_cols = X.columns[X.isna().any()].tolist()
    print(f"Dropping columns with NaN or Infinity values: {invalid_cols}")
    X_cleaned = X.drop(columns=invalid_cols)
    
    return X_cleaned

def build_features(X, y):
    """Perform feature engineering and apply SMOTE to handle class imbalance using GPU acceleration."""
    
    # Define paths
    vis_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\src\visualization"
    processed_data_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\processed"
    
    # Create directories if they don't exist
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)

    # Drop columns with NaN or infinity values
    X_cleaned = drop_invalid_columns(X)

    # Save class distribution before SMOTE
    plot_class_distribution(y, 'Class Distribution Before SMOTE', os.path.join(vis_path, 'class_distribution_before_smote.png'))
    
    # Convert X to GPU tensor for faster operations
    X_cleaned_cp = cp.asarray(X_cleaned)
    y_cp = cp.asarray(y)

    # 1. Polynomial Features: Create interaction terms and higher-degree features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(cp.asnumpy(X_cleaned_cp))
    poly_feature_names = poly.get_feature_names_out(X_cleaned.columns)
    X_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
    
    # 2. Binning: Apply binning to continuous variables like 'Amount'
    binning = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    X_poly['Amount_binned'] = binning.fit_transform(X_poly[['Amount']])

    # 3. Log Transformation: Apply log transformation to 'Amount' and other skewed features
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    X_poly['log_Amount'] = log_transformer.transform(np.clip(X_poly[['Amount']].values, 0, None))[:, 0]

    # 4. Dimensionality Reduction: Use PCA to reduce the feature space while retaining variance
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_reduced = pd.DataFrame(pca.fit_transform(X_poly), columns=[f'PC{i}' for i in range(pca.n_components_)])

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reduced, y)

    # Plot class distribution after SMOTE
    plot_class_distribution(y_resampled, 'Class Distribution After SMOTE', os.path.join(vis_path, 'class_distribution_after_smote.png'))
    
    print(f"Original dataset shape: {X_cleaned.shape}, {y.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}, {y_resampled.shape}")
    
    # Save the feature-engineered data
    X_resampled.to_csv(os.path.join(processed_data_path, 'X_features_resampled.csv'), index=False)
    pd.DataFrame(y_resampled, columns=['y']).to_csv(os.path.join(processed_data_path, 'y_resampled.csv'), index=False)
    
    print(f"Feature-engineered data saved to {os.path.join(processed_data_path, 'X_features_resampled.csv')}")
    print(f"Resampled target data saved to {os.path.join(processed_data_path, 'y_resampled.csv')}")
    
    return X_resampled, y_resampled

if __name__ == "__main__":
    # Load preprocessed data
    X_scaled, y = load_preprocessed_data()
    
    # Build new features and apply SMOTE
    X_resampled, y_resampled = build_features(X_scaled, y)
