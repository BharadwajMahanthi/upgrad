import pandas as pd
import numpy as np
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from src.config import VISUALIZATION_DIR, RANDOM_STATE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for Feature Engineering.
    Includes:
    - Interaction Terms
    - Polynomial Features
    - Log Transformations
    """
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.poly_feature_names = None
        self.columns_to_poly = ['x43', 'x44', 'x45']

    def fit(self, X, y=None):
        # Save feature names for robustness against numpy conversion
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        
        # Fit Polynomial Features on specified columns
        if all(col in X.columns for col in self.columns_to_poly):
            self.poly.fit(X[self.columns_to_poly])
            self.poly_feature_names = self.poly.get_feature_names_out(self.columns_to_poly)
        else:
            logging.warning(f"Columns {self.columns_to_poly} not found for Polynomial fitting.")
        return self

    def transform(self, X):
        X_eng = X.copy()
        
        # Defensive: Handle numpy arrays if pipeline loses dataframe context
        if not hasattr(X_eng, "columns"):
            logging.info("CustomFeatureEngineer: Input missing columns (likely numpy) - Fixing...")
            try:
                # Use fitted feature names if available (Robust Fix)
                if hasattr(self, "feature_names_in_") and len(self.feature_names_in_) == X_eng.shape[1]:
                    cols = self.feature_names_in_
                    logging.info("  Restoring columns from fitted feature_names_in_.")
                else:
                    # Fallback (risky but better than crash)
                    cols = [f"x{i+1}" for i in range(X_eng.shape[1])]
                    logging.warning("  Using generic x1...xN columns (feature names mismatch or missing).")
                
                X_eng = pd.DataFrame(X_eng, columns=cols)
            except Exception as e:
                logging.error(f"  Failed to convert numpy to DataFrame: {e}")
                raise e
        
        # 1. Interaction Terms
        if 'x43' in X_eng.columns and 'x44' in X_eng.columns:
            X_eng['interaction_term_x43_x44'] = X_eng['x43'] * X_eng['x44']
        if 'x45' in X_eng.columns and 'x46' in X_eng.columns:
            X_eng['interaction_term_x45_x46'] = X_eng['x45'] * X_eng['x46']
        
        # 2. Polynomial Features
        if self.poly_feature_names is not None:
            poly_features = self.poly.transform(X_eng[self.columns_to_poly])
            X_poly = pd.DataFrame(poly_features, columns=list(self.poly_feature_names), index=X_eng.index)
            X_eng = pd.concat([X_eng, X_poly], axis=1)
        
        # 3. Log Transformation
        # Using log1p to handle zeros/small values safely
        transformer = FunctionTransformer(np.log1p, validate=False)
        for col in ['x43', 'x44']:
            if col in X_eng.columns:
                # Clip negative values to 0 before log to avoid NaNs
                # transform returns a 2D array, we need to flatten it
                transformed_values = transformer.transform(np.clip(X_eng[[col]].values, 0, None))
                if isinstance(transformed_values, np.ndarray):
                     X_eng[f'log_{col}'] = transformed_values[:, 0]

        
        # Handle infinities created by transformations
        X_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return X_eng

def apply_smote(X, y):
    """
    Skipping SMOTE as per user request.
    Just performing Median Imputation to ensure clean data for training.
    """
    logging.info(f"SMOTE Disabled. Performing Imputation only. Original shape: {X.shape}")
    
    # We must handle NaNs 
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Return original counts (no resampling)
    logging.info(f"Data prepared (No SMOTE). Shape: {X_imputed.shape}")
    
    return X_imputed, y
