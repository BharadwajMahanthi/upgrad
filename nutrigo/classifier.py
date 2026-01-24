# %%
import os
import re
import json
import ast
import joblib
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import class_weight

# TensorFlow
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# NLTK
import nltk
from nltk.corpus import stopwords

# Surprise
from surprise import Dataset as SurpriseDataset
from surprise import Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split

warnings.filterwarnings("ignore")

# --------------------------- Configuration --------------------------- #
CORE_RECIPE_PATH = Path("core-data_recipe.csv")
RAW_RECIPE_PATH = Path("raw-data_recipe.csv")
CORE_TRAIN_PATH = Path("core-data-train_rating.csv")
CORE_VALID_PATH = Path("core-data-valid_rating.csv")
CORE_TEST_PATH = Path("core-data-test_rating.csv")
RAW_INTERACTION_PATH = Path("raw-data_interaction.csv")

CORE_IMAGE_DIR = Path("core-data-images")
RAW_IMAGE_DIR = Path("raw-data-images")

CORE_IMAGE_FEATURES_PATH = Path("core_image_features.npy")
RAW_IMAGE_FEATURES_PATH = Path("raw_image_features.npy")
TFIDF_VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")
SCALER_PATH = Path("scaler.pkl")
HEALTH_MODEL_PATH = Path("healthiness_model.h5")
LE_USER_PATH = Path("le_user.pkl")
LE_RECIPE_PATH = Path("le_recipe.pkl")
SVD_MODEL_PATH = Path("svd_model.pkl")

TARGET_NUTRIENTS = ["calories", "protein", "fat", "carbohydrates", "fiber"]
IMAGE_BATCH_SIZE = 32
LOG_LEVEL = logging.INFO

# --------------------------- Logging --------------------------- #
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


# --------------------------- Reproducibility --------------------------- #
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds(42)


# --------------------------- TensorFlow GPU Configuration --------------------------- #
def configure_tensorflow():
    """
    Configure GPU memory growth if GPU exists. Choose best strategy automatically.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logging.info("No GPU detected. TensorFlow will run on CPU.")
        return tf.distribute.get_strategy()

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"GPU(s) detected: {len(gpus)}")
    except RuntimeError as e:
        logging.warning(f"GPU memory growth setup failed: {e}")

    # Multi-GPU → MirroredStrategy, else default strategy
    if len(gpus) > 1:
        logging.info("Using MirroredStrategy (multi-GPU).")
        return tf.distribute.MirroredStrategy()

    logging.info("Using default strategy (single GPU).")
    return tf.distribute.get_strategy()


STRATEGY = configure_tensorflow()


# --------------------------- Fast Text Preprocessing --------------------------- #
_TEXT_CLEAN_RE = re.compile(r"[^a-zA-Z\s]+")


def init_nltk():
    """
    Downloads required NLTK data once (if missing).
    """
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    # punkt is not strictly necessary if we avoid word_tokenize,
    # but keeping it in case you extend later.
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


init_nltk()
STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Faster preprocessing:
    - lowercase
    - remove non alphabet
    - split on whitespace (fast)
    - remove stopwords
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _TEXT_CLEAN_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


# --------------------------- Nutrition Parsing --------------------------- #
def _to_float(x):
    """
    Convert "123g", "12.5", "$20" → float safely.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        # keep only digits and dot
        cleaned = re.sub(r"[^\d.]", "", x)
        return float(cleaned) if cleaned else np.nan

    return np.nan


def safe_parse_dict(s):
    """
    Try JSON then Python literal dict.
    """
    if not isinstance(s, str) or not s.strip():
        return {}

    # common cleanup
    s2 = s.strip()

    # First try JSON
    try:
        # Some datasets store single quotes → make it json-friendly
        if "'" in s2 and '"' not in s2:
            s2_json = s2.replace("'", '"')
        else:
            s2_json = s2
        return json.loads(s2_json)
    except Exception:
        pass

    # Fallback: literal_eval
    try:
        return ast.literal_eval(s2)
    except Exception:
        return {}


def parse_nutritions_nested(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse nutrition dict column into numeric columns.
    Supports nested format like:
      {"calories":{"amount":"120"}, ...}
    """
    if "nutritions" not in df.columns:
        raise ValueError("Expected 'nutritions' column but not found.")

    logging.info("Parsing 'nutritions' column...")
    parsed = df["nutritions"].apply(safe_parse_dict)

    def extract_one(d):
        out = {}
        for n in TARGET_NUTRIENTS:
            v = d.get(n, np.nan)

            if isinstance(v, dict):
                out[n] = _to_float(v.get("amount", np.nan))
            else:
                out[n] = _to_float(v)

        return pd.Series(out)

    nutritions_df = parsed.apply(extract_one)

    # Fill NaNs with mean per column
    nutritions_df = nutritions_df.apply(pd.to_numeric, errors="coerce")
    nutritions_df = nutritions_df.fillna(nutritions_df.mean(numeric_only=True))

    df = df.drop(columns=["nutritions"]).reset_index(drop=True)
    df = pd.concat([df, nutritions_df], axis=1)

    # Final numeric coercion safety
    for c in TARGET_NUTRIENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET_NUTRIENTS] = df[TARGET_NUTRIENTS].fillna(df[TARGET_NUTRIENTS].mean())

    logging.info("Nutrition parsing completed.")
    return df


# --------------------------- Image Feature Extraction --------------------------- #
def load_and_preprocess_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img


def extract_image_features(
    recipes_df: pd.DataFrame,
    image_dir: Path,
    output_file: Path,
    feature_model: tf.keras.Model,
    batch_size: int = 32,
) -> np.ndarray:
    logging.info(f"Extracting image features from: {image_dir}")

    image_paths = []
    valid_indices = []

    for idx, recipe_id in enumerate(recipes_df["recipe_id"].astype(str).tolist()):
        img_path = image_dir / f"{recipe_id}.jpg"
        if img_path.exists():
            image_paths.append(str(img_path))
            valid_indices.append(idx)

    # If no images exist, return zeros safely
    feature_dim = feature_model.output_shape[-1]
    full_features = np.zeros((len(recipes_df), feature_dim), dtype=np.float32)

    if len(image_paths) == 0:
        logging.warning(f"No images found in {image_dir}. Saving zero features.")
        np.save(output_file, full_features)
        return full_features

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    extracted = []
    for batch in tqdm(dataset, desc=f"Extracting ({image_dir.name})"):
        feats = feature_model(batch, training=False).numpy()
        extracted.append(feats)

    extracted = np.vstack(extracted).astype(np.float32)
    full_features[np.array(valid_indices)] = extracted

    np.save(output_file, full_features)
    logging.info(f"Saved image features: {output_file}")
    return full_features


def load_or_extract_image_features(
    recipes_df: pd.DataFrame,
    image_dir: Path,
    output_file: Path,
    feature_model: tf.keras.Model,
    batch_size: int = 32,
) -> np.ndarray:
    if output_file.exists():
        logging.info(f"Loading image features from: {output_file}")
        return np.load(output_file).astype(np.float32)

    return extract_image_features(
        recipes_df=recipes_df,
        image_dir=image_dir,
        output_file=output_file,
        feature_model=feature_model,
        batch_size=batch_size,
    )


# --------------------------- Data Loading --------------------------- #
def assert_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def load_data():
    for p in [CORE_RECIPE_PATH, RAW_RECIPE_PATH, CORE_TRAIN_PATH, CORE_VALID_PATH, CORE_TEST_PATH, RAW_INTERACTION_PATH]:
        assert_exists(p)

    logging.info("Loading CSV files...")
    core_recipes = pd.read_csv(CORE_RECIPE_PATH)
    raw_recipes = pd.read_csv(RAW_RECIPE_PATH, low_memory=False)

    core_train = pd.read_csv(CORE_TRAIN_PATH)
    core_valid = pd.read_csv(CORE_VALID_PATH)
    core_test = pd.read_csv(CORE_TEST_PATH)
    raw_interactions = pd.read_csv(RAW_INTERACTION_PATH)

    logging.info(f"Core recipes shape: {core_recipes.shape}")
    logging.info(f"Raw recipes shape: {raw_recipes.shape}")

    return core_recipes, raw_recipes, core_train, core_valid, core_test, raw_interactions


# --------------------------- Preprocessing --------------------------- #
def preprocess_data(core_recipes, raw_recipes, core_train, core_valid, core_test, raw_interactions):
    logging.info("Cleaning missing values...")

    # Drop missing core/raw recipe essentials
    core_recipes = core_recipes.dropna(subset=["recipe_id", "recipe_name", "ingredients", "nutritions"]).copy()
    raw_recipes = raw_recipes.dropna(subset=["recipe_id", "recipe_name", "ingredients", "nutritions"]).copy()

    # Drop missing interaction rows
    core_train = core_train.dropna().copy()
    core_valid = core_valid.dropna().copy()
    core_test = core_test.dropna().copy()
    raw_interactions = raw_interactions.dropna().copy()

    # Convert IDs to string
    for df in [core_train, core_valid, core_test, raw_interactions]:
        df["user_id"] = df["user_id"].astype(str)
        df["recipe_id"] = df["recipe_id"].astype(str)

    core_recipes["recipe_id"] = core_recipes["recipe_id"].astype(str)
    raw_recipes["recipe_id"] = raw_recipes["recipe_id"].astype(str)

    logging.info("Fitting encoders...")
    le_user = LabelEncoder()
    le_recipe = LabelEncoder()

    combined_user_ids = pd.concat(
        [core_train["user_id"], core_valid["user_id"], core_test["user_id"], raw_interactions["user_id"]]
    ).drop_duplicates()
    le_user.fit(combined_user_ids)

    combined_recipe_ids = pd.concat([core_recipes["recipe_id"], raw_recipes["recipe_id"]]).drop_duplicates()
    le_recipe.fit(combined_recipe_ids)

    for df in [core_train, core_valid, core_test, raw_interactions]:
        df["user_id_encoded"] = le_user.transform(df["user_id"])
        df["recipe_id_encoded"] = le_recipe.transform(df["recipe_id"])

    core_recipes["recipe_id_encoded"] = le_recipe.transform(core_recipes["recipe_id"])
    raw_recipes["recipe_id_encoded"] = le_recipe.transform(raw_recipes["recipe_id"])

    joblib.dump(le_user, LE_USER_PATH)
    joblib.dump(le_recipe, LE_RECIPE_PATH)
    logging.info("Saved label encoders.")

    logging.info("Preprocessing ingredients text...")
    core_recipes["ingredients_clean"] = core_recipes["ingredients"].apply(preprocess_text)
    raw_recipes["ingredients_clean"] = raw_recipes["ingredients"].apply(preprocess_text)

    # Parse nutritions
    core_recipes = parse_nutritions_nested(core_recipes)
    raw_recipes = parse_nutritions_nested(raw_recipes)

    return core_recipes, raw_recipes, core_train, core_valid, core_test, raw_interactions, le_user, le_recipe


# --------------------------- Feature Engineering --------------------------- #
def feature_engineering(core_recipes, raw_recipes):
    logging.info("TF-IDF vectorizing ingredients...")

    if TFIDF_VECTORIZER_PATH.exists():
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        ingredient_tfidf_core = tfidf_vectorizer.transform(core_recipes["ingredients_clean"])
        ingredient_tfidf_raw = tfidf_vectorizer.transform(raw_recipes["ingredients_clean"])
    else:
        tfidf_vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        ingredient_tfidf_core = tfidf_vectorizer.fit_transform(core_recipes["ingredients_clean"])
        ingredient_tfidf_raw = tfidf_vectorizer.transform(raw_recipes["ingredients_clean"])
        joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
        logging.info(f"Saved TF-IDF vectorizer: {TFIDF_VECTORIZER_PATH}")

    logging.info("Scaling nutrient features...")
    core_recipes[TARGET_NUTRIENTS] = core_recipes[TARGET_NUTRIENTS].apply(pd.to_numeric, errors="coerce")
    raw_recipes[TARGET_NUTRIENTS] = raw_recipes[TARGET_NUTRIENTS].apply(pd.to_numeric, errors="coerce")

    core_recipes[TARGET_NUTRIENTS] = core_recipes[TARGET_NUTRIENTS].fillna(core_recipes[TARGET_NUTRIENTS].mean())
    raw_recipes[TARGET_NUTRIENTS] = raw_recipes[TARGET_NUTRIENTS].fillna(raw_recipes[TARGET_NUTRIENTS].mean())

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        core_scaled = scaler.transform(core_recipes[TARGET_NUTRIENTS])
        raw_scaled = scaler.transform(raw_recipes[TARGET_NUTRIENTS])
    else:
        scaler = StandardScaler()
        core_scaled = scaler.fit_transform(core_recipes[TARGET_NUTRIENTS])
        raw_scaled = scaler.transform(raw_recipes[TARGET_NUTRIENTS])
        joblib.dump(scaler, SCALER_PATH)
        logging.info(f"Saved scaler: {SCALER_PATH}")

    for i, col in enumerate(TARGET_NUTRIENTS):
        core_recipes[f"{col}_scaled"] = core_scaled[:, i]
        raw_recipes[f"{col}_scaled"] = raw_scaled[:, i]

    logging.info("Creating image feature extractor model...")

    with STRATEGY.scope():
        feature_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",  # returns vector (1280,)
            input_shape=(224, 224, 3),
        )
        feature_model.trainable = False  # pure feature extraction

    core_image_features = load_or_extract_image_features(
        core_recipes, CORE_IMAGE_DIR, CORE_IMAGE_FEATURES_PATH, feature_model, IMAGE_BATCH_SIZE
    )
    raw_image_features = load_or_extract_image_features(
        raw_recipes, RAW_IMAGE_DIR, RAW_IMAGE_FEATURES_PATH, feature_model, IMAGE_BATCH_SIZE
    )

    return (
        ingredient_tfidf_core,
        ingredient_tfidf_raw,
        core_image_features,
        raw_image_features,
        tfidf_vectorizer,
        scaler,
    )


# --------------------------- SVD Recommender --------------------------- #
def build_and_evaluate_svd(core_train, core_valid, core_test):
    logging.info("Building and evaluating SVD recommender...")

    all_core_interactions = pd.concat([core_train, core_valid, core_test], ignore_index=True)

    # Surprise works fine with integers here
    min_rating = float(all_core_interactions["rating"].min())
    max_rating = float(all_core_interactions["rating"].max())

    reader = Reader(rating_scale=(min_rating, max_rating))
    data = SurpriseDataset.load_from_df(
        all_core_interactions[["user_id_encoded", "recipe_id_encoded", "rating"]],
        reader,
    )

    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

    svd_model = SVD(
        n_factors=100,
        n_epochs=25,
        lr_all=0.005,
        reg_all=0.2,
        random_state=42,
    )

    svd_model.fit(trainset)
    predictions = svd_model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=True)

    joblib.dump(svd_model, SVD_MODEL_PATH)
    logging.info(f"Saved SVD model: {SVD_MODEL_PATH}")
    logging.info(f"SVD RMSE: {rmse:.4f}")

    return svd_model, rmse


# --------------------------- Healthiness Model --------------------------- #
def build_healthiness_model(core_recipes: pd.DataFrame):
    logging.info("Training healthiness model...")

    # Example metric: healthy if calories < 500
    core_recipes = core_recipes.copy()
    core_recipes["is_healthy"] = (core_recipes["calories"] < 500).astype(int)

    feature_cols = [f"{c}_scaled" for c in TARGET_NUTRIENTS]
    X = core_recipes[feature_cols].values.astype(np.float32)
    y = core_recipes["is_healthy"].values.astype(np.int32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # class weights
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = {int(k): float(v) for k, v in zip(np.unique(y_train), cw)}

    with STRATEGY.scope():
        model = Sequential(
            [
                InputLayer(input_shape=(X_train.shape[1],)),
                Dense(128, activation="relu"),
                Dropout(0.35),
                Dense(64, activation="relu"),
                Dropout(0.25),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    early_stopping = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=256,
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Health model test accuracy: {acc:.4f}")

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    logging.info(f"ROC AUC: {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Unhealthy", "Healthy"])
    plt.yticks([0, 1], ["Unhealthy", "Healthy"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig("healthiness_confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("healthiness_roc_curve.png")
    plt.close()

    # Training curves
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_accuracy.png")
    plt.close()

    model.save(HEALTH_MODEL_PATH)
    logging.info(f"Saved health model: {HEALTH_MODEL_PATH}")
    return model


# --------------------------- Main --------------------------- #
def main():
    try:
        core_recipes, raw_recipes, core_train, core_valid, core_test, raw_interactions = load_data()

        (
            core_recipes,
            raw_recipes,
            core_train,
            core_valid,
            core_test,
            raw_interactions,
            le_user,
            le_recipe,
        ) = preprocess_data(core_recipes, raw_recipes, core_train, core_valid, core_test, raw_interactions)

        (
            ingredient_tfidf_core,
            ingredient_tfidf_raw,
            core_image_features,
            raw_image_features,
            tfidf_vectorizer,
            scaler,
        ) = feature_engineering(core_recipes, raw_recipes)

        svd_model, svd_rmse = build_and_evaluate_svd(core_train, core_valid, core_test)
        print(f"\n✅ SVD RMSE: {svd_rmse:.4f}")

        health_model = build_healthiness_model(core_recipes)
        print("\n✅ All tasks completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline crashed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
