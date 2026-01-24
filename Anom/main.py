import sys
import logging
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Add the source directory to the system path to allow imports
# This is necessary because 'src' is not installed as a package in this environment
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import src.config as config
from src.data.load_data import load_and_clean_data, perform_initial_eda, perform_advanced_eda
from src.features.build_features import CustomFeatureEngineer, apply_smote
import src.models.train_model as train_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Force sklearn to output pandas DataFrames instead of numpy arrays
# This preserves column names for the CustomFeatureEngineer
try:
    set_config(transform_output="pandas")
except Exception:
    logging.warning("Could not set sklearn transform_output to pandas. Check sklearn version >= 1.2")

def main():
    config.setup_directories()
    logging.info("Pipeline Started.")
    
    # ----------------------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------------------
    try:
        data = load_and_clean_data(config.RAW_DATA_PATH)
        # Drop y to create X, isolate y for target
        X = data.drop(columns=['y'])
        y = data['y']
    except FileNotFoundError as e:
        logging.error(f"{e} - Please make sure data is in {config.RAW_DATA_PATH}")
        return

    # Initial EDA
    top_features = perform_initial_eda(data)
    perform_advanced_eda(data, top_features) # STRONG EDA

    # ----------------------------------------------------------------
    # 2. Split Data (CRITICAL STEP to prevent leakage)
    # ----------------------------------------------------------------
    logging.info("Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y # Important for imbalanced data
    )
    logging.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # ----------------------------------------------------------------
    # 3. Define Preprocessing Pipeline
    # ----------------------------------------------------------------
    # This pipeline handles operations that must be FIT on Train and TRANSFORM on Test
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Median is robust to outliers
        ('feature_engineering', CustomFeatureEngineer()),
        ('scaler', StandardScaler())
    ])

    # ----------------------------------------------------------------
    # 4. Process Training Data
    # ----------------------------------------------------------------
    logging.info("Preprocessing Training Data...")
    
    # Fit and Transform Train
    try:
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
    except TypeError as e:
        # Fallback for older sklearn versions where set_config doesn't work effectively for all transformers
        logging.error(f"Pipeline failed: {e}. Attempting manual column recovery...")
        raise
    
    # ----------------------------------------------------------------
    # 5. Handle Imbalance (SMOTE) - ONLY ON TRAIN
    # ----------------------------------------------------------------
    # SMOTE cannot be easily inside standard Pipeline (needs imblearn pipeline).
    # Applying it manually here on the processed training data.
    
    # Check if we have columns (Pandas) or not (Numpy)
    if not isinstance(X_train_processed, pd.DataFrame):
        logging.info("Pipeline returned Numpy Int array. Converting to DataFrame for SMOTE/Model compatibility if needed.")
        # Try to reconstruct dataframe using transformed feature names if possible, else generic
        # This is tricky. For now, we will proceed.
        pass

    X_train_resampled, y_train_resampled = apply_smote(X_train_processed, y_train)

    # ----------------------------------------------------------------
    # 6. Preprocess Test Data (BEFORE model comparison)
    # ----------------------------------------------------------------
    logging.info("Preprocessing Test Data...")
    X_test_processed = preprocessor.transform(X_test)
    
    # ----------------------------------------------------------------
    # 7. Model Comparison & Auto-Selection
    # ----------------------------------------------------------------
    logging.info("Comparing Models and Auto-Selecting Best...")
    comparison_df, best_model, best_model_name = train_model.compare_models(
        X_train_resampled, y_train_resampled, X_test_processed, y_test
    )
    
    # ----------------------------------------------------------------
    # 8. Fine-Tune Best Model (if RandomForest or GradientBoosting)
    # ----------------------------------------------------------------
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        logging.info(f"Fine-tuning {best_model_name} with RandomizedSearchCV...")
        
        if best_model_name == 'Random Forest':
            search_results, cv_results = train_model.train_rf_model(X_train_resampled, y_train_resampled)
            best_model = search_results.best_estimator_
        else:
            # For Gradient Boosting, use similar hyperparameter search
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.ensemble import GradientBoostingClassifier
            from scipy.stats import randint as sp_randint
            from scipy.stats import uniform as sp_uniform
            
            param_dist = {
                'n_estimators': sp_randint(50, 200),
                'max_depth': sp_randint(3, 10),
                'learning_rate': sp_uniform(0.01, 0.3),
                'subsample': sp_uniform(0.6, 0.4),
                'min_samples_split': sp_randint(2, 10)
            }
            
            search = RandomizedSearchCV(
                GradientBoostingClassifier(random_state=config.RANDOM_STATE),
                param_distributions=param_dist,
                n_iter=20,
                cv=3,
                scoring='f1',
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
            search.fit(X_train_resampled, y_train_resampled)
            best_model = search.best_estimator_
            logging.info(f"Best Params: {search.best_params_}")
    else:
        logging.info(f"{best_model_name} selected - using default configuration.")
    
    # ----------------------------------------------------------------
    # 9. Evaluate on Test Data
    # ----------------------------------------------------------------
    # Note: evaluate_model calculates and returns the optimal threshold
    best_threshold = train_model.evaluate_model(best_model, X_test_processed, y_test, X_train_resampled, y_train_resampled)
    
    # ----------------------------------------------------------------
    # 10. Save Full Inference Pipeline
    # ----------------------------------------------------------------
    logging.info("Constructing and Saving Full Inference Pipeline...")
    
    # Create a pipeline that includes the fitted preprocessor and the fitted model
    # This ensures raw data (59 features) can be passed to the API
    inference_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    
    # Save bundle
    import joblib
    bundle = {
        "model": inference_pipeline,
        "threshold": best_threshold,
        "feature_names": list(X.columns) # Save original feature names for validation
    }
    joblib.dump(bundle, config.MODEL_SAVE_PATH)
    logging.info(f"✅ Full Pipeline saved to {config.MODEL_SAVE_PATH} with threshold={best_threshold:.2f}")

    logging.info("Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()
