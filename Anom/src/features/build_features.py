import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_preprocessed_data():
    """Load preprocessed data from the processed folder."""
    X_scaled = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\processed\X_scaled.csv')
    y = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\processed\y.csv')
    
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

def plot_distribution_before_after(X_before, X_after, columns, title, save_path):
    """Plot distributions before and after feature engineering."""
    plt.figure(figsize=(12, 8))
    for col in columns:
        sns.kdeplot(X_before[col], label=f'Before: {col}', fill=True)
        sns.kdeplot(X_after[col], label=f'After: {col}', fill=True)
    plt.title(title)
    plt.legend()
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
    """Perform feature engineering and apply SMOTE to handle class imbalance."""
    
    # Define paths
    vis_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\anom\src\visualization"
    processed_data_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\processed"
    
    # Create directories if they don't exist
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Handle missing values before SMOTE
    imputer = SimpleImputer(strategy='mean')  # Use mean imputation to fill missing values
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Drop columns with NaN or infinity values
    X_cleaned = drop_invalid_columns(X_imputed)

    # Save class distribution before SMOTE
    plot_class_distribution(y, 'Class Distribution Before SMOTE', os.path.join(vis_path, 'class_distribution_before_smote.png'))
    
    # Copy of X before feature engineering
    X_before = X_cleaned.copy()

    # 1. Interaction Terms: Create interaction terms between certain pairs of features
    X_cleaned['interaction_term_x43_x44'] = X_cleaned['x43'] * X_cleaned['x44']
    X_cleaned['interaction_term_x45_x46'] = X_cleaned['x45'] * X_cleaned['x46']
    
    # 2. Polynomial Features: Create polynomial features from certain columns (e.g., x43, x44, x45)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X_cleaned[['x43', 'x44', 'x45']])  # Example on selective columns
    poly_feature_names = poly.get_feature_names_out(['x43', 'x44', 'x45'])
    X_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
    
    # Add the polynomial features to the main DataFrame
    X_cleaned = pd.concat([X_cleaned, X_poly], axis=1)
    
    # 3. Log Transformation: Apply log transformation to certain features (e.g., x43, x44)
    transformer = FunctionTransformer(np.log1p, validate=True)  # log1p to handle zeros

    # Ensure that we only transform non-negative values
    X_cleaned['log_x43'] = transformer.transform(np.clip(X_cleaned[['x43']].values, 0, None))[:, 0]
    X_cleaned['log_x44'] = transformer.transform(np.clip(X_cleaned[['x44']].values, 0, None))[:, 0]

    # Plot distributions of some features before and after transformation
    plot_distribution_before_after(X_before, X_cleaned, ['x43', 'x44'], 'Feature Distributions Before and After Log Transformation', 
                                   os.path.join(vis_path, 'feature_distribution_before_after_log_transform.png'))
    
    # Drop columns with NaN or infinity values again after transformation
    X_cleaned = drop_invalid_columns(X_cleaned)
    
    # 4. Apply SMOTE to balance the classes (after handling missing values)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_cleaned, y)
    
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
