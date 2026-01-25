import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Data directory
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Models and Artifacts directory
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Dataset file paths
CORE_RECIPE_PATH = DATA_DIR / "core-data_recipe.csv"
RAW_RECIPE_PATH = DATA_DIR / "raw-data_recipe.csv"
CORE_TRAIN_PATH = DATA_DIR / "core-data-train_rating.csv"
CORE_VALID_PATH = DATA_DIR / "core-data-valid_rating.csv"
CORE_TEST_PATH = DATA_DIR / "core-data-test_rating.csv"
RAW_INTERACTION_PATH = DATA_DIR / "raw-data_interaction.csv"

# Image directories
CORE_IMAGE_DIR = DATA_DIR / "core-data-images"
RAW_IMAGE_DIR = DATA_DIR / "raw-data-images"

# Model and artifact paths
CORE_IMAGE_FEATURES_PATH = MODELS_DIR / "core_image_features.npy"
RAW_IMAGE_FEATURES_PATH = MODELS_DIR / "raw_image_features.npy"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
TEXT_SVD_PATH = MODELS_DIR / "text_svd.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
HEALTH_MODEL_PATH = MODELS_DIR / "healthiness_model.keras"
HEALTH_SCALER_PATH = MODELS_DIR / "health_scaler.pkl"
LE_USER_PATH = MODELS_DIR / "le_user.pkl"
LE_RECIPE_PATH = MODELS_DIR / "le_recipe.pkl"
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"

# Hybrid engine cache paths
RECIPE_MASTER_CACHE_PATH = MODELS_DIR / "recipe_master_cache.pkl"
RECIPE_EMBEDDINGS_PATH = MODELS_DIR / "recipe_embeddings.npy"
NEIGHBORS_INDEX_PATH = MODELS_DIR / "neighbors_index.pkl"

# Database path
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATA_DIR / 'nutrigo.db'}"

# Nutritional columns
TARGET_NUTRIENTS = ["calories", "protein", "fat", "carbohydrates", "fiber"]

# Recommendation config
TFIDF_MAX_FEATURES = 2000
TEXT_EMBED_DIM = 256
CANDIDATES_FROM_CONTENT = 800
