import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src import config
from src.data.load_data import perform_eda, preprocess_data
from src.features.build_features import load_preprocessed_data, build_features
from src.models.train_model import train_logistic_regression_torch

def main():
    # --------------------------------------------------------
    # Step 1: EDA and Preprocessing
    # --------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 1: Load Data & Preprocessing")
    print("=" * 50)
    perform_eda(config.RAW_DATA_FILE)
    
    # Preprocess (Scale Time/Amount)
    # Returns X_scaled, y, preprocessed_df
    X_scaled, y, _ = preprocess_data(config.CLEANED_DATA_FILE)

    # --------------------------------------------------------
    # Step 2: Feature Engineering
    # --------------------------------------------------------
    # Note: load_preprocessed_data is redundant since we have X_scaled, y from step 1
    # but we will stick to the function flow for modularity if needed.
    # We can just pass the in-memory variables.
    
    print("\n" + "=" * 50)
    print("STEP 2: Feature Engineering & SMOTE")
    print("=" * 50)
    
    # build_features saves to disk AND returns the resampled sets
    # Explicitly cast to numpy array to satisfy type checker
    X_resampled, y_resampled = build_features(X_scaled, np.asarray(y))

    # --------------------------------------------------------
    # Step 3: Model Training
    # --------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 3: Model Training")
    print("=" * 50)

    # Hyperparameters
    param_distributions = {
        'lr': [0.001, 0.01, 0.1], 
        'num_epochs': [150, 300, 600]
    }
    regularization = 'l1'
    reg_lambda = 0.01
    
    # Model path prefers torch extension
    model_path = config.BEST_TORCH_MODEL_FILE if hasattr(config, "BEST_TORCH_MODEL_FILE") else Path(config.BEST_MODEL_FILE)

    train_logistic_regression_torch(
        X_resampled, 
        y_resampled, 
        param_distributions=param_distributions, 
        n_iter=5, 
        regularization=regularization, 
        reg_lambda=reg_lambda, 
        model_save_path=model_path
    )
    
    print("\n" + "=" * 50)
    print("✅ Pipeline Completed Successfully")
    print("=" * 50)

if __name__ == "__main__":
    main()
