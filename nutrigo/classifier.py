# classification_engine.py
"""
Healthiness Classification Module (Separated)

This file contains ONLY classification logic.
It is designed to work alongside:
  - app.py (Flask)
  - hybrid_engine.py (Hybrid recommender)

Classification goal:
  Predict whether a recipe is healthy based on nutrition columns.

Artifacts saved:
  - healthiness_model.h5
  - health_scaler.pkl
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import config

# --------------------------- CONFIG --------------------------- #
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)

# Paths
HEALTH_MODEL_PATH = config.HEALTH_MODEL_PATH
HEALTH_SCALER_PATH = config.HEALTH_SCALER_PATH

# Default dataset file (optional training)
CORE_RECIPE_PATH = config.CORE_RECIPE_PATH

TARGET_NUTRIENTS = config.TARGET_NUTRIENTS


# --------------------------- GPU/CPU SETUP --------------------------- #
def configure_tensorflow() -> tf.distribute.Strategy:
    """
    Configure TensorFlow GPU memory growth if GPU exists.
    Returns strategy for training.
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

    if len(gpus) > 1:
        logging.info("Using MirroredStrategy (multi-GPU).")
        return tf.distribute.MirroredStrategy()

    logging.info("Using default strategy (single GPU).")
    return tf.distribute.get_strategy()


STRATEGY = configure_tensorflow()


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds(42)


# --------------------------- UTILS --------------------------- #
def parse_nutritions_nested(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'nutritions' column with nested dictionary structure.
    Used when training standalone from CSV.
    """
    if "nutritions" not in df.columns:
        return df

    def clean_nutrition_str(nutrition_str):
        if isinstance(nutrition_str, str):
            return nutrition_str.replace("u'", "'")
        return "{}"

    cleaned_nutritions = df["nutritions"].apply(clean_nutrition_str)

    def safe_literal_eval(nutrition_str):
        try:
            return ast.literal_eval(nutrition_str)
        except Exception:
            return {}

    nutritions_expanded = cleaned_nutritions.apply(safe_literal_eval)

    def extract_nutrients(nutrition_dict):
        nutrient_values = {}
        for nutrient in TARGET_NUTRIENTS:
            if nutrient in nutrition_dict:
                nutrition_info = nutrition_dict[nutrient]
                if isinstance(nutrition_info, dict):
                    amount = nutrition_info.get("amount", np.nan)
                    if isinstance(amount, str):
                        amount = "".join(filter(lambda c: c.isdigit() or c == ".", amount))
                    nutrient_values[nutrient] = float(amount) if amount else np.nan
                elif isinstance(nutrition_info, (int, float)):
                    nutrient_values[nutrient] = float(nutrition_info)
                else:
                    nutrient_values[nutrient] = np.nan
            else:
                nutrient_values[nutrient] = np.nan
        return pd.Series(nutrient_values)

    nutritions_df = nutritions_expanded.apply(extract_nutrients)
    nutritions_df = nutritions_df.fillna(nutritions_df.mean(numeric_only=True))

    df = pd.concat([df, nutritions_df], axis=1)
    df.drop("nutritions", axis=1, inplace=True)
    return df


def ensure_nutrient_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures nutrient columns exist and are numeric.
    Missing columns are filled with NaN.
    """
    df = df.copy()
    
    # If nutrients are still in JSON format, parse them
    if "nutritions" in df.columns and any(df[TARGET_NUTRIENTS[0]].isna() if TARGET_NUTRIENTS[0] in df.columns else [True]):
        df = parse_nutritions_nested(df)

    for col in TARGET_NUTRIENTS:
        if col not in df.columns:
            df[col] = np.nan

    for col in TARGET_NUTRIENTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values with mean
    df[TARGET_NUTRIENTS] = df[TARGET_NUTRIENTS].fillna(df[TARGET_NUTRIENTS].mean(numeric_only=True))

    return df


def default_health_label(df: pd.DataFrame, calorie_threshold: float = 500.0) -> np.ndarray:
    """
    Rule-based label used to train the classifier:
      healthy if calories < 500
    """
    if "calories" not in df.columns:
        raise ValueError("calories column is required to create labels")
    return (df["calories"].astype(float) < float(calorie_threshold)).astype(int).values


# --------------------------- CLASSIFIER --------------------------- #
class HealthinessClassifier:
    """
    A standalone classifier for healthiness prediction.

    Usage:
      clf = HealthinessClassifier()
      clf.load_or_train(df)
      prob = clf.predict_health_prob(df_single_row)
      is_healthy = clf.predict_is_healthy(df_single_row)
    """

    def __init__(
        self,
        model_path: Path = HEALTH_MODEL_PATH,
        scaler_path: Path = HEALTH_SCALER_PATH,
        calorie_threshold: float = 500.0,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.calorie_threshold = float(calorie_threshold)

        self.model: tf.keras.Model | None = None
        self.scaler: StandardScaler | None = None

    # -------------------- TRAINING -------------------- #
    def fit_scaler(self, df: pd.DataFrame) -> StandardScaler:
        df = ensure_nutrient_columns(df)
        scaler = StandardScaler()
        scaler.fit(df[TARGET_NUTRIENTS].values)
        self.scaler = scaler
        return scaler

    def build_model(self, input_dim: int) -> tf.keras.Model:
        """
        Small dense network (fast for Flask deployments).
        """
        with STRATEGY.scope():
            model = Sequential(
                [
                    InputLayer(input_shape=(input_dim,)),
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

        return model

    def train(
        self,
        df: pd.DataFrame,
        save_artifacts: bool = True,
        plot_metrics: bool = True,
    ) -> tf.keras.Model:
        """
        Train classifier from a dataframe containing nutrients.
        Creates labels using rule: calories < threshold.
        """
        df = ensure_nutrient_columns(df)
        y = default_health_label(df, calorie_threshold=self.calorie_threshold).astype(np.int32)

        # Fit scaler
        scaler = self.fit_scaler(df)
        X = scaler.transform(df[TARGET_NUTRIENTS].values).astype(np.float32)

        # Split sets
        if len(np.unique(y)) < 2:
            raise ValueError("Not enough label variety to train. Only one class found.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # Class weights
        cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights_dict = {int(k): float(v) for k, v in zip(np.unique(y_train), cw)}
        logging.info(f"Class weights: {class_weights_dict}")

        # Build + train model
        model = self.build_model(input_dim=X_train.shape[1])

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

        print("\nClassification Report (Healthiness):")
        print(classification_report(y_test, y_pred))

        roc_auc = roc_auc_score(y_test, y_prob)
        logging.info(f"ROC AUC: {roc_auc:.4f}")

        # Save plots
        if plot_metrics:
            self._save_training_plots(y_test, y_pred, y_prob, roc_auc, history)

        # Save artifacts
        if save_artifacts:
            self.save(model=model, scaler=scaler)

        self.model = model
        return model

    def _save_training_plots(self, y_test, y_pred, y_prob, roc_auc, history):
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title("Confusion Matrix (Healthiness)")
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
        plt.title("ROC Curve (Healthiness)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig("healthiness_roc_curve.png")
        plt.close()

        # Training history
        if history and hasattr(history, "history"):
            plt.figure(figsize=(10, 4))
            plt.plot(history.history.get("accuracy", []), label="train_acc")
            plt.plot(history.history.get("val_accuracy", []), label="val_acc")
            plt.title("Training Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig("healthiness_training_accuracy.png")
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.plot(history.history.get("loss", []), label="train_loss")
            plt.plot(history.history.get("val_loss", []), label="val_loss")
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig("healthiness_training_loss.png")
            plt.close()

    # -------------------- SAVE/LOAD -------------------- #
    def save(self, model: tf.keras.Model, scaler: StandardScaler) -> None:
        model.save(self.model_path)
        joblib.dump(scaler, self.scaler_path)
        logging.info(f"Saved model to: {self.model_path}")
        logging.info(f"Saved scaler to: {self.scaler_path}")

    def load(self) -> bool:
        """
        Load model+scaler if they exist.
        Returns True if loaded successfully.
        """
        if not self.model_path.exists() or not self.scaler_path.exists():
            return False

        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        logging.info(f"Loaded model from: {self.model_path}")
        logging.info(f"Loaded scaler from: {self.scaler_path}")
        return True

    def load_or_train(self, df: pd.DataFrame) -> tf.keras.Model:
        """
        Load artifacts if possible. Otherwise train from df.
        """
        loaded = self.load()
        if loaded:
            return self.model
        return self.train(df)

    # -------------------- INFERENCE -------------------- #
    def _ensure_ready(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Classifier is not ready. Call load() or load_or_train() first.")

    def predict_health_prob(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability that recipe is healthy.
        df can be a single-row df or batch df.
        """
        self._ensure_ready()
        df = ensure_nutrient_columns(df)
        X = self.scaler.transform(df[TARGET_NUTRIENTS].values).astype(np.float32)
        probs = self.model.predict(X, verbose=0).flatten()
        return probs

    def predict_is_healthy(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_health_prob(df)
        return (probs >= float(threshold)).astype(int)

    def predict_recipe_row(
        self,
        calories: float,
        protein: float,
        fat: float,
        carbohydrates: float,
        fiber: float,
        threshold: float = 0.5,
    ) -> dict:
        """
        Convenient prediction for single recipe from raw numbers.
        """
        df = pd.DataFrame(
            [{
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbohydrates": carbohydrates,
                "fiber": fiber
            }]
        )

        prob = float(self.predict_health_prob(df)[0])
        return {"health_prob": prob, "is_healthy": int(prob >= threshold)}


# --------------------------- OPTIONAL TRAINING FROM CSV --------------------------- #
def train_from_csv(csv_path: Path = CORE_RECIPE_PATH) -> HealthinessClassifier:
    """
    Optional utility: train classification model directly from CSV.
    The CSV must include TARGET_NUTRIENTS columns already parsed.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {Path(csv_path).resolve()}")

    df = pd.read_csv(csv_path)

    # If the CSV has nutritions JSON column, your app.py already parses it.
    # Here we assume nutrients exist already.
    df = ensure_nutrient_columns(df)

    clf = HealthinessClassifier()
    clf.train(df, save_artifacts=True, plot_metrics=True)
    return clf


# --------------------------- CLI MAIN --------------------------- #
if __name__ == "__main__":
    """
    Run:
      python classification_engine.py

    This will train from core-data_recipe.csv if model isn't present.
    """
    try:
        df = pd.read_csv(CORE_RECIPE_PATH)
        df = ensure_nutrient_columns(df)

        clf = HealthinessClassifier()
        if clf.load():
            logging.info("✅ Classifier already trained and loaded.")
        else:
            logging.info("Training classifier from CSV...")
            clf.train(df, save_artifacts=True, plot_metrics=True)
            logging.info("✅ Training completed.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
