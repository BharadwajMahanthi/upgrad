from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
import logging
import joblib
import pandas as pd
import numpy as np
import ast
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from surprise import SVD, Dataset, Reader
import os
from datetime import datetime
from flask_migrate import Migrate
import config

# NEW: ML utilities for hybrid engine
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# NEW: Import healthiness classifier
from classifier import HealthinessClassifier

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure key

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Increase busy timeout for SQLite (5 minutes)
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "connect_args": {"timeout": 300}
}
db = SQLAlchemy(app)

# Enable WAL mode for better concurrency
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    # Only apply to sqlite connections
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

migrate = Migrate(app, db)

# -------------------------- GLOBAL MODELS -------------------------- #
svd_model = None
le_user = None
le_recipe = None

# NEW: Hybrid recommender global engine
hybrid_engine = None

# NEW: Healthiness classifier global
health_classifier = HealthinessClassifier()

# -------------------------- FILE PATHS -------------------------- #
CORE_RECIPE_PATH = config.CORE_RECIPE_PATH
RAW_RECIPE_PATH = config.RAW_RECIPE_PATH
CORE_TRAIN_PATH = config.CORE_TRAIN_PATH
CORE_VALID_PATH = config.CORE_VALID_PATH
CORE_TEST_PATH = config.CORE_TEST_PATH
RAW_INTERACTION_PATH = config.RAW_INTERACTION_PATH

CORE_IMAGE_DIR = config.CORE_IMAGE_DIR
RAW_IMAGE_DIR = config.RAW_IMAGE_DIR

CORE_IMAGE_FEATURES_PATH = config.CORE_IMAGE_FEATURES_PATH
RAW_IMAGE_FEATURES_PATH = config.RAW_IMAGE_FEATURES_PATH

TFIDF_VECTORIZER_PATH = config.TFIDF_VECTORIZER_PATH
TEXT_SVD_PATH = config.TEXT_SVD_PATH
SCALER_PATH = config.SCALER_PATH

RECIPE_MASTER_CACHE_PATH = config.RECIPE_MASTER_CACHE_PATH
RECIPE_EMBEDDINGS_PATH = config.RECIPE_EMBEDDINGS_PATH
NEIGHBORS_INDEX_PATH = config.NEIGHBORS_INDEX_PATH

HEALTH_MODEL_PATH = config.HEALTH_MODEL_PATH
LE_USER_PATH = config.LE_USER_PATH
LE_RECIPE_PATH = config.LE_RECIPE_PATH
SVD_MODEL_PATH = config.SVD_MODEL_PATH

# Define nutritional columns
TARGET_NUTRIENTS = config.TARGET_NUTRIENTS

# Hybrid config
TFIDF_MAX_FEATURES = config.TFIDF_MAX_FEATURES
TEXT_EMBED_DIM = config.TEXT_EMBED_DIM
CANDIDATES_FROM_CONTENT = config.CANDIDATES_FROM_CONTENT


# -------------------------- TEXT PREPROCESS -------------------------- #
_TEXT_CLEAN_RE = re.compile(r"[^a-zA-Z\s]+")


def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _TEXT_CLEAN_RE.sub(" ", text)
    return " ".join(text.split())


def tokenize_ingredients(clean_text: str):
    if not isinstance(clean_text, str) or not clean_text.strip():
        return set()
    return set(clean_text.split())


# -------------------------- Nutrition Parser -------------------------- #
def parse_nutritions_nested(df):
    logging.info("Parsing 'nutritions' column with nested dictionary structure...")

    def clean_nutrition_str(nutrition_str):
        if isinstance(nutrition_str, str):
            cleaned_str = nutrition_str.replace("u'", "'")
            return cleaned_str
        return "{}"

    cleaned_nutritions = df["nutritions"].apply(clean_nutrition_str)

    def safe_literal_eval(nutrition_str):
        try:
            return ast.literal_eval(nutrition_str)
        except Exception as e:
            logging.error(f"Error parsing nutritions: {e} for string: {nutrition_str}")
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
    nutritions_df = nutritions_df.fillna(nutritions_df.mean())

    df = pd.concat([df, nutritions_df], axis=1)
    df.drop("nutritions", axis=1, inplace=True)

    for col in TARGET_NUTRIENTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_NUTRIENTS] = df[TARGET_NUTRIENTS].fillna(df[TARGET_NUTRIENTS].mean())
    logging.info("Parsing 'nutritions' column completed.")
    return df


def clean_cooking_directions(raw_dir):
    """
    Parses the dictionary-like string from dataset: {'directions': u'Step 1...'}
    """
    if pd.isnull(raw_dir):
        return ""
    raw_dir_str = str(raw_dir).strip()
    if raw_dir_str.startswith("{"):
        try:
            # Safely evaluate JSON-like dict string
            eval_str = raw_dir_str.replace("u'", "'")
            d = ast.literal_eval(eval_str)
            return d.get("directions", "").strip()
        except Exception:
            return raw_dir_str
    return raw_dir_str


def fix_image_url(url, recipe_id=None):
    """
    Standardize image URLs: Prioritize local images matching <recipe_id>.jpg,
    otherwise upgrade to HTTPS and handle dead domains.
    """
    if recipe_id:
        # Check local directories first
        recipe_id_str = str(recipe_id)
        filename = f"{recipe_id_str}.jpg"
        
        core_path = os.path.join(config.CORE_IMAGE_DIR, filename)
        raw_path = os.path.join(config.RAW_IMAGE_DIR, filename)
        
        if os.path.exists(core_path) or os.path.exists(raw_path):
            return f"/recipe_images/{filename}"

    if not url or not isinstance(url, str):
        return None
        
    url = url.strip()
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)
    
    return url


def ensure_data_integrity():
    """
    One-time migration check to ensure instructions and ingredients are cleaned in the SQL DB.
    """
    try:
        logging.info("Checking database integrity (instructions & ingredients)...")
        # Check if we have any records with ^ or literal "None" in instructions
        recipes_to_fix = Recipe.query.filter(
            (Recipe.ingredients.like('%^%')) | 
            (Recipe.instructions == None) | 
            (Recipe.instructions == 'None')
        ).limit(1000).all()

        if recipes_to_fix:
            logging.info(f"Found recipes needing optimization. Applying fixes...")
            # We'll reload from CSV for instructions if they are missing
            csv_path = config.CORE_RECIPE_PATH
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, usecols=["recipe_id", "cooking_directions"])
                df["recipe_id"] = df["recipe_id"].astype(str)
                lookup = df.set_index("recipe_id")["cooking_directions"].to_dict()
                
                for r in recipes_to_fix:
                    # Fix ingredients
                    if r.ingredients and "^" in r.ingredients:
                        r.ingredients = r.ingredients.replace("^", "\n")
                    
                    # Fix instructions
                    if not r.instructions or r.instructions == "None":
                        raw_dir = lookup.get(str(r.recipe_id))
                        if raw_dir:
                            r.instructions = clean_cooking_directions(raw_dir)
                    
                    # Fix image URLs while we are here
                    r.image_url = fix_image_url(r.image_url, r.recipe_id)

                db.session.commit()
                logging.info("Database integrity migration completed successfully.")
    except Exception as e:
        logging.error(f"Integrity check failed: {e}")
        db.session.rollback()


# -------------------------- DB Models -------------------------- #
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), unique=True, nullable=False)
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    height = db.Column(db.Float)
    goal = db.Column(db.String(100))
    health_details = db.Column(db.Text)
    preferences = db.Column(db.Text)
    # ✅ DB already has this
    password_hash = db.Column(db.String(128))
    interactions = relationship("Interaction", back_populates="user")


class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.String(50), unique=True, nullable=False)
    recipe_name = db.Column(db.String(255))
    ingredients = db.Column(db.Text)
    # NOTE: dataset has cooking_directions -> stored as instructions
    instructions = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    calories = db.Column(db.Float)
    protein = db.Column(db.Float)
    fat = db.Column(db.Float)
    carbohydrates = db.Column(db.Float)
    fiber = db.Column(db.Float)
    sodium = db.Column(db.Float)
    sugars = db.Column(db.Float)
    cholesterol = db.Column(db.Float)
    saturated_fat = db.Column(db.Float)
    rating = db.Column(db.Float, default=0.0)
    review_nums = db.Column(db.Integer, default=0)
    interactions = relationship("Interaction", back_populates="recipe")
    reviews = relationship("Review", back_populates="recipe")


class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), db.ForeignKey("user.user_id"))
    recipe_id = db.Column(db.String(50), db.ForeignKey("recipe.recipe_id"))
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
    user = relationship("User", back_populates="interactions")
    recipe = relationship("Recipe", back_populates="interactions")


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.String(50), db.ForeignKey("recipe.recipe_id"))
    user_id = db.Column(db.String(50))
    rating = db.Column(db.Float)
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime)
    recipe = relationship("Recipe", back_populates="reviews")


# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
ADMIN_PASSWORD_HASH = generate_password_hash(ADMIN_PASSWORD)


# -------------------------- HYBRID ENGINE -------------------------- #
# HybridRecommender is now imported from hybrid_engine.py
from hybrid_engine import HybridRecommender


def build_or_load_hybrid_engine(force_rebuild=False):
    """
    Builds or loads Hybrid Engine from DB recipes:
      embedding = [TextSVD(256) + scaled_nutrition(5)]
    """
    global hybrid_engine

    # If cached files exist and not forcing rebuild -> load
    if (
        (not force_rebuild)
        and os.path.exists(RECIPE_MASTER_CACHE_PATH)
        and os.path.exists(RECIPE_EMBEDDINGS_PATH)
        and os.path.exists(NEIGHBORS_INDEX_PATH)
        and os.path.exists(TFIDF_VECTORIZER_PATH)
        and os.path.exists(TEXT_SVD_PATH)
        and os.path.exists(SCALER_PATH)
    ):
        try:
            recipe_master = joblib.load(RECIPE_MASTER_CACHE_PATH)
            embeddings = np.load(RECIPE_EMBEDDINGS_PATH).astype(np.float32)
            neighbors = joblib.load(NEIGHBORS_INDEX_PATH)
            tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
            text_svd = joblib.load(TEXT_SVD_PATH)
            scaler = joblib.load(SCALER_PATH)

            hybrid_engine = HybridRecommender(recipe_master, embeddings, neighbors, tfidf, text_svd, scaler)
            logging.info("Hybrid Engine loaded from cache ✅")
            return hybrid_engine
        except Exception as e:
            logging.warning(f"Failed loading hybrid cache. Rebuilding... {e}")

    # Build from DB
    with app.app_context():
        recipes = Recipe.query.all()

    if not recipes:
        logging.warning("No recipes found in DB. Hybrid Engine cannot be built.")
        hybrid_engine = None
        return None

    recipe_master = pd.DataFrame(
        [
            {
                "recipe_id": str(r.recipe_id),
                "recipe_name": r.recipe_name,
                "ingredients": r.ingredients,
                "ingredients_clean": preprocess_text(r.ingredients),
                "calories": float(r.calories) if r.calories is not None else np.nan,
                "protein": float(r.protein) if r.protein is not None else np.nan,
                "fat": float(r.fat) if r.fat is not None else np.nan,
                "carbohydrates": float(r.carbohydrates) if r.carbohydrates is not None else np.nan,
                "fiber": float(r.fiber) if r.fiber is not None else np.nan,
            }
            for r in recipes
        ]
    )

    # Fill nutrients
    for c in TARGET_NUTRIENTS:
        recipe_master[c] = pd.to_numeric(recipe_master[c], errors="coerce")
    recipe_master[TARGET_NUTRIENTS] = recipe_master[TARGET_NUTRIENTS].fillna(recipe_master[TARGET_NUTRIENTS].mean())

    # TF-IDF
    if os.path.exists(TFIDF_VECTORIZER_PATH) and not force_rebuild:
        tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
        tfidf_mat = tfidf.transform(recipe_master["ingredients_clean"])
    else:
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
        tfidf_mat = tfidf.fit_transform(recipe_master["ingredients_clean"])
        joblib.dump(tfidf, TFIDF_VECTORIZER_PATH)

    # Text SVD
    if os.path.exists(TEXT_SVD_PATH) and not force_rebuild:
        text_svd = joblib.load(TEXT_SVD_PATH)
        text_emb = text_svd.transform(tfidf_mat).astype(np.float32)
    else:
        text_svd = TruncatedSVD(n_components=TEXT_EMBED_DIM, random_state=42)
        text_emb = text_svd.fit_transform(tfidf_mat).astype(np.float32)
        joblib.dump(text_svd, TEXT_SVD_PATH)

    # Nutrition scaler
    if os.path.exists(SCALER_PATH) and not force_rebuild:
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = StandardScaler()
        scaler.fit(recipe_master[TARGET_NUTRIENTS].values)
        joblib.dump(scaler, SCALER_PATH)

    nutr_scaled = scaler.transform(recipe_master[TARGET_NUTRIENTS].values).astype(np.float32)

    # Final embedding = [text_emb + nutr_scaled]
    embeddings = np.hstack([text_emb, nutr_scaled]).astype(np.float32)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms

    # Neighbors index
    neighbors = NearestNeighbors(metric="cosine", algorithm="auto", n_neighbors=min(2000, len(recipe_master)))
    neighbors.fit(embeddings)

    # Save caches
    joblib.dump(recipe_master, RECIPE_MASTER_CACHE_PATH)
    np.save(RECIPE_EMBEDDINGS_PATH, embeddings)
    joblib.dump(neighbors, NEIGHBORS_INDEX_PATH)

    hybrid_engine = HybridRecommender(recipe_master, embeddings, neighbors, tfidf, text_svd, scaler)
    logging.info("Hybrid Engine built and cached ✅")
    return hybrid_engine


# -------------------------- DB POPULATION -------------------------- #
def populate_database():
    recipe_count = Recipe.query.count()
    interaction_count = Interaction.query.count()

    if recipe_count == 0:
        logging.info("Populating database from CSV files...")

        if os.path.exists(CORE_RECIPE_PATH):
            recipes_df = pd.read_csv(CORE_RECIPE_PATH)
            recipes_df = parse_nutritions_nested(recipes_df)
            recipes_df["recipe_id"] = recipes_df["recipe_id"].astype(str)

            expected_columns = [
                "recipe_id",
                "recipe_name",
                "ingredients",
                "cooking_directions",  # ✅ REAL COLUMN in dataset
                "image_url",
                "calories",
                "protein",
                "fat",
                "carbohydrates",
                "fiber",
            ]
            missing_columns = set(expected_columns) - set(recipes_df.columns)
            if missing_columns:
                logging.warning(f"Missing columns in recipes CSV: {missing_columns}")
                for col in missing_columns:
                    recipes_df[col] = None

            for _, row in recipes_df.iterrows():
                # PERMANENT FIX: Clean ingredients and instructions during population
                raw_ingredients = str(row.get("ingredients", ""))
                clean_ingredients = raw_ingredients.replace("^", "\n")

                recipe = Recipe(
                    recipe_id=str(row["recipe_id"]),
                    recipe_name=row.get("recipe_name"),
                    ingredients=clean_ingredients,
                    instructions=clean_cooking_directions(row.get("cooking_directions")),
                    image_url=fix_image_url(row.get("image_url")),
                    calories=row.get("calories"),
                    protein=row.get("protein"),
                    fat=row.get("fat"),
                    carbohydrates=row.get("carbohydrates"),
                    fiber=row.get("fiber"),
                    rating=None,
                )
                db.session.add(recipe)

            db.session.commit()
            logging.info("Recipes data populated.")
        else:
            logging.error(f"Recipe CSV file '{CORE_RECIPE_PATH}' not found.")
            return

    if interaction_count == 0:
        interaction_file_path = CORE_TRAIN_PATH
        if os.path.exists(interaction_file_path):
            logging.info("Populating interactions from CSV file...")
            interactions_df = pd.read_csv(interaction_file_path)
            interactions_df["user_id"] = interactions_df["user_id"].astype(str)
            interactions_df["recipe_id"] = interactions_df["recipe_id"].astype(str)

            user_ids = interactions_df["user_id"].unique()
            for user_id in user_ids:
                existing_user = User.query.filter_by(user_id=user_id).first()
                if not existing_user:
                    user = User(user_id=user_id)
                    db.session.add(user)

            for _, row in interactions_df.iterrows():
                ts_raw = row.get("dateLastModified")  # ✅ Use dateLastModified
                try:
                    ts_raw = str(ts_raw).strip()
                    ts = pd.to_datetime(ts_raw, errors="coerce")
                    if pd.isnull(ts):
                        ts = None
                except Exception:
                    ts = None

                interaction = Interaction(
                    user_id=str(row["user_id"]),
                    recipe_id=str(row["recipe_id"]),
                    rating=float(row["rating"]) if pd.notnull(row["rating"]) else None,
                    timestamp=ts,
                )
                db.session.add(interaction)

            db.session.commit()
            logging.info("Interactions data populated.")
        else:
            logging.warning(
                f"Interactions CSV file '{interaction_file_path}' not found. Skipping interactions data population."
            )
    else:
        logging.info("Database already populated.")


# -------------------------- MODEL TRAINING -------------------------- #
def retrain_model():
    global svd_model, le_user, le_recipe, hybrid_engine

    logging.info("Retraining the recommendation model...")

    interactions = Interaction.query.all()
    if not interactions:
        logging.warning("No interaction data available for model training.")
        svd_model = None
        return

    data = []
    for interaction in interactions:
        if interaction.rating is None:
            continue
        data.append((str(interaction.user_id), str(interaction.recipe_id), float(interaction.rating)))

    df = pd.DataFrame(data, columns=["user_id", "recipe_id", "rating"])

    # Save encoders (optional but kept for compatibility)
    from sklearn.preprocessing import LabelEncoder

    le_user = LabelEncoder()
    le_recipe = LabelEncoder()
    le_user.fit(df["user_id"].astype(str))
    le_recipe.fit(df["recipe_id"].astype(str))

    joblib.dump(le_user, LE_USER_PATH)
    joblib.dump(le_recipe, LE_RECIPE_PATH)

    # Surprise trained on RAW IDs (important)
    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["user_id", "recipe_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    svd_model = SVD()
    svd_model.fit(trainset)

    joblib.dump(svd_model, SVD_MODEL_PATH)
    logging.info("Model retraining completed ✅")

    # Rebuild hybrid engine so it stays synced with DB
    try:
        build_or_load_hybrid_engine(force_rebuild=True)
    except Exception as e:
        logging.warning(f"Could not rebuild hybrid engine after retrain: {e}")


def initialize_app():
    global svd_model, le_user, le_recipe, hybrid_engine, health_classifier

    # Extract db path from URL (remove 'sqlite:///')
    db_path = str(config.DATA_DIR / "nutrigo.db")
    
    if not os.path.exists(db_path):
        logging.info(f"Database file '{db_path}' not found. Creating a new database.")
        with app.app_context():
            db.create_all()
            populate_database()
            retrain_model()
    else:
        logging.info(f"Database file '{db_path}' already exists. Skipping database creation and population.")
        with app.app_context():
            # ✅ NEW: Permanent migration check
            ensure_data_integrity()
            
            if os.path.exists(SVD_MODEL_PATH):
                try:
                    svd_model = joblib.load(SVD_MODEL_PATH)
                    logging.info("SVD model loaded successfully.")
                except Exception as e:
                    logging.warning(f"Could not load SVD model: {e}")
                    svd_model = None

            if os.path.exists(LE_USER_PATH) and os.path.exists(LE_RECIPE_PATH):
                try:
                    le_user = joblib.load(LE_USER_PATH)
                    le_recipe = joblib.load(LE_RECIPE_PATH)
                    logging.info("Label encoders loaded successfully.")
                except Exception as e:
                    logging.warning(f"Could not load label encoders: {e}")
                    le_user, le_recipe = None, None

            # Load/Build health classifier
            try:
                if health_classifier.load():
                    logging.info("Healthiness Classifier loaded successfully.")
                else:
                    logging.info("Healthiness Classifier model not found. It will be trained if needed.")
            except Exception as e:
                logging.warning(f"Could not load health classifier: {e}")

            # Build/load hybrid engine
            try:
                build_or_load_hybrid_engine(force_rebuild=False)
            except Exception as e:
                logging.warning(f"Hybrid engine not available: {e}")
                hybrid_engine = None

            # If no SVD model, retrain
            if svd_model is None:
                logging.info("No trained SVD model found. Retraining now...")
                retrain_model()


initialize_app()


# -------------------------- ADMIN -------------------------- #
def admin_required(f):
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip().lower()
        password = (request.form.get("password") or "").strip()

        logging.info(f"Admin login attempt for username: {username}")

        # Ensure comparison is robust
        target_admin = ADMIN_USERNAME.strip().lower()

        if username == target_admin and (check_password_hash(ADMIN_PASSWORD_HASH, password) or password == ADMIN_PASSWORD):
            session["admin_logged_in"] = True
            flash("Logged in as admin.")
            logging.info("Admin login successful.")
            return redirect(url_for("admin_dashboard"))
        else:
            logging.warning(f"Admin login failed for user: {username}")
            flash("Invalid admin credentials.")
            return redirect(url_for("admin_login"))
    return render_template("admin_login.html")


@app.route("/admin/logout")
@admin_required
def admin_logout():
    session.pop("admin_logged_in", None)
    flash("Logged out successfully.")
    return redirect(url_for("home"))


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    users = User.query.all()
    return render_template("admin_dashboard.html", users=users)


@app.route("/admin/edit_user/<int:user_id>", methods=["GET", "POST"])
@admin_required
def admin_edit_user(user_id):
    target_user = User.query.get_or_404(user_id)
    if request.method == "POST":
        age = request.form.get("age")
        weight = request.form.get("weight")
        height = request.form.get("height")
        goal = request.form.get("goal")
        health_details = request.form.get("health_details")
        preferences = request.form.get("preferences")
        new_password = request.form.get("new_password")

        # Update biological data
        if age:
            target_user.age = int(age)
        if weight:
            target_user.weight = float(weight)
        if height:
            target_user.height = float(height)
        if goal:
            target_user.goal = goal
        if health_details:
            target_user.health_details = health_details
        if preferences:
            target_user.preferences = preferences

        # Password reset if provided
        if new_password and new_password.strip():
            target_user.password_hash = generate_password_hash(new_password.strip())
            logging.info(f"Admin reset password for user: {target_user.user_id}")

        db.session.commit()
        flash(f"User {target_user.user_id} updated successfully.")
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_edit_user.html", target_user=target_user)


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if user:
        Interaction.query.filter_by(user_id=user.user_id).delete()
        Review.query.filter_by(user_id=user.user_id).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f"User {user.user_id} deleted successfully.")
    else:
        flash("User not found.")
    return redirect(url_for("admin_dashboard"))


# -------------------------- RECIPE DETAIL + REVIEWS -------------------------- #
@app.route("/recipe/<recipe_id>", methods=["GET", "POST"])
def recipe_detail(recipe_id):
    recipe = Recipe.query.filter_by(recipe_id=recipe_id).first()
    if not recipe:
        flash("Recipe not found.")
        return redirect(url_for("home"))

    # CLEAN INGREDIENTS: Split by newline (now permanent in DB)
    ingredients_list = []
    if recipe.ingredients:
        # Handle the new newline format
        ingredients_list = [i.strip() for i in str(recipe.ingredients).split('\n') if i.strip()]

    if request.method == "POST":
        user_id = request.form.get("user_id")
        rating = request.form.get("rating")
        comment = request.form.get("comment")

        review = Review(
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            rating=float(rating),
            comment=comment,
            timestamp=datetime.utcnow(),
        )
        db.session.add(review)
        db.session.commit()
        flash("Your review has been submitted.")

    reviews = Review.query.filter_by(recipe_id=recipe_id).order_by(Review.timestamp.desc()).all()
    # Ensure image URL is fixed (local vs HTTPS)
    recipe.image_url = fix_image_url(recipe.image_url, recipe.recipe_id)
    
    return render_template("recipe_detail.html", recipe=recipe, reviews=reviews, ingredients_list=ingredients_list)


# -------------------------- CLASSIFICATION -------------------------- #
@app.route("/classify/<recipe_id>", methods=["GET"])
def classify_recipe(recipe_id):
    recipe = Recipe.query.filter_by(recipe_id=str(recipe_id)).first()
    if not recipe:
        flash("Recipe not found.")
        return redirect(url_for("home"))

    df = pd.DataFrame(
        [
            {
                "calories": recipe.calories,
                "protein": recipe.protein,
                "fat": recipe.fat,
                "carbohydrates": recipe.carbohydrates,
                "fiber": recipe.fiber,
            }
        ]
    )

    is_healthy = False
    prob = None

    if health_classifier.model is not None:
        try:
            prob = float(health_classifier.predict_health_prob(df)[0])
            is_healthy = bool(prob >= 0.5)
        except Exception as e:
            logging.error(f"Classification failed: {e}")
            # fallback
            if recipe.calories is not None:
                is_healthy = float(recipe.calories) < 500
                prob = 1.0 if is_healthy else 0.0
    else:
        # fallback rule-based classification if model not trained
        if recipe.calories is not None:
            is_healthy = float(recipe.calories) < 500
            prob = 1.0 if is_healthy else 0.0

    # Return HTML template
    return render_template("classification.html", recipe=recipe, is_healthy=is_healthy, probability=prob)


# -------------------------- RECOMMENDATION FUNCTION -------------------------- #
def get_recommendations(user_id, preferences_input=None, n_recommendations=10):
    """
    Uses Hybrid Engine if available.
    Falls back to SVD-only recommendation if hybrid not available.
    Supports:
      - keyword cold start if preferences_input provided
      - constraints from user.goal
      - excludes already seen recipes
    """
    global hybrid_engine

    user = User.query.filter_by(user_id=str(user_id)).first()
    if not user:
        logging.warning(f"User ID {user_id} not found in database.")
        return []

    preferences = user.preferences
    goal = user.goal

    # Combine preferences from profile + input
    combined_keywords = ""
    if preferences:
        combined_keywords += str(preferences)
    if preferences_input:
        if combined_keywords:
            combined_keywords += ", " + str(preferences_input)
        else:
            combined_keywords = str(preferences_input)

    # Auto constraints based on goal
    constraints = {}
    if goal:
        g = goal.lower()
        if "diet" in g:
            constraints["max_calories"] = 450
        if "muscle" in g:
            constraints["min_protein"] = 20

    # ---------------- HYBRID PATH ---------------- #
    if hybrid_engine is not None:
        try:
            # If keyword search provided -> cold-start from keywords
            if combined_keywords.strip():
                recs_df = hybrid_engine.recommend_from_keywords(
                    keywords=combined_keywords,
                    top_k=n_recommendations,
                    constraints=constraints if constraints else None,
                )
            else:
                recs_df = hybrid_engine.recommend_hybrid(
                    user_id=str(user_id),
                    svd_model=svd_model,
                    top_k=n_recommendations,
                    alpha=0.70,
                    constraints=constraints if constraints else None,
                    exclude_seen=True,
                )

            if recs_df.empty:
                return []

            recommendations = []
            for _, row in recs_df.iterrows():
                recipe = Recipe.query.filter_by(recipe_id=str(row["recipe_id"])).first()
                if not recipe:
                    continue

                calories = "N/A"
                protein = "N/A"
                if recipe.calories is not None and not pd.isnull(recipe.calories):
                    calories = int(recipe.calories)
                if recipe.protein is not None and not pd.isnull(recipe.protein):
                    protein = int(recipe.protein)

                image_url = fix_image_url(recipe.image_url, recipe.recipe_id)

                recommendations.append(
                    {
                        "recipe_id": recipe.recipe_id,
                        "recipe_name": recipe.recipe_name,
                        "calories": calories,
                        "image_url": image_url,
                        "protein": protein,
                        "rating": round(float(recipe.rating), 1) if recipe.rating else 0.0,
                        "review_nums": recipe.review_nums or 0,
                        "est_rating": round(float(row["svd_score"]), 2) if not pd.isna(row["svd_score"]) else 0.0,
                        "final_score": round(float(row["final_score"]), 4) if "final_score" in row else None,
                    }
                )

            return recommendations

        except Exception as e:
            logging.error(f"Hybrid recommendation failed: {e}")

    # ---------------- FALLBACK: SVD ONLY ---------------- #
    logging.info("Hybrid engine not available. Using fallback SVD-only recommendations...")

    recipes = Recipe.query.all()
    recipes_df = pd.DataFrame(
        [
            {
                "recipe_id": recipe.recipe_id,
                "recipe_name": recipe.recipe_name,
                "ingredients": recipe.ingredients,
                "image_url": recipe.image_url,
                "calories": recipe.calories,
                "protein": recipe.protein,
            }
            for recipe in recipes
        ]
    )

    if svd_model:
        all_recipe_ids = recipes_df["recipe_id"].astype(str).tolist()

        predictions = []
        for recipe_id in all_recipe_ids:
            try:
                pred = svd_model.predict(str(user_id), str(recipe_id))
                predictions.append((recipe_id, float(pred.est)))
            except Exception:
                predictions.append((recipe_id, 0.0))

        pred_df = pd.DataFrame(predictions, columns=["recipe_id", "est_rating"])
        recommendations_df = pd.merge(pred_df, recipes_df, on="recipe_id")
        recommendations_df = recommendations_df.sort_values(by="est_rating", ascending=False)
    else:
        recommendations_df = recipes_df.copy()
        recommendations_df["est_rating"] = 0.0

    top_recommendations = recommendations_df.head(n_recommendations)

    recommendations = []
    for _, row in top_recommendations.iterrows():
        calories = row.get("calories", "N/A")
        if pd.isnull(calories):
            calories = "N/A"
        else:
            calories = int(calories)

        protein = row.get("protein", "N/A")
        if pd.isnull(protein):
            protein = "N/A"
        else:
            protein = int(protein)

        image_url = row.get("image_url", "")
        if pd.isnull(image_url) or str(image_url).strip() == "":
            image_url = None

        recommendations.append(
            {
                "recipe_id": row["recipe_id"],
                "recipe_name": row["recipe_name"],
                "calories": calories,
                "image_url": fix_image_url(image_url, row["recipe_id"]),
                "est_rating": round(float(row.get("est_rating", 0)), 2),
                "rating": round(float(row.get("rating", 0)), 1) if not pd.isnull(row.get("rating")) else 0.0,
                "review_nums": int(row.get("review_nums", 0)) if not pd.isnull(row.get("review_nums")) else 0,
                "protein": protein,
                "final_score": None,
            }
        )

    return recommendations


# -------------------------- ROUTES -------------------------- #
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user_id = request.form.get("user_id").strip()

        existing_user = User.query.filter_by(user_id=user_id).first()
        if existing_user:
            flash("User ID already exists. Please choose a different one.")
            return redirect(url_for("register"))

        age = request.form.get("age")
        weight = request.form.get("weight")
        height = request.form.get("height")
        goal = request.form.get("goal")
        health_details = request.form.get("health_details")
        preferences = request.form.get("preferences")

        new_user = User(
            user_id=user_id,
            age=int(age) if age else None,
            weight=float(weight) if weight else None,
            height=float(height) if height else None,
            goal=goal,
            health_details=health_details,
            preferences=preferences,
        )

        db.session.add(new_user)
        db.session.commit()

        flash(f"Registration successful! Your User ID is {user_id}. Please use it to log in.")
        return redirect(url_for("home"))
    return render_template("register.html")


@app.route("/update_profile", methods=["GET", "POST"])
def update_profile():
    if request.method == "POST":
        user_id = request.form.get("user_id").strip()
        user = User.query.filter_by(user_id=user_id).first()
        if not user:
            flash("User ID not found.")
            return redirect(url_for("update_profile"))

        age = request.form.get("age")
        weight = request.form.get("weight")
        height = request.form.get("height")
        goal = request.form.get("goal")
        health_details = request.form.get("health_details")
        preferences = request.form.get("preferences")

        if age:
            user.age = int(age)
        if weight:
            user.weight = float(weight)
        if height:
            user.height = float(height)
        if goal:
            user.goal = goal
        if health_details:
            user.health_details = health_details
        if preferences:
            user.preferences = preferences

        db.session.commit()
        flash("Profile updated successfully.")
        return redirect(url_for("home"))

    return render_template("update_profile.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id").strip()
    preferences_input = request.form.get("preferences", "").strip()

    logging.info(f"Received recommendation request for User ID: {user_id}")

    recommendations = get_recommendations(user_id, preferences_input)

    if not recommendations:
        error_message = "No recommendations found. Please check your User ID or register if you are a new user."
        return render_template("recommendations.html", error_message=error_message)

    return render_template("recommendations.html", recommendations=recommendations)


# ✅ NEW: Explain Route (works with template OR JSON)
@app.route("/explain/<recipe_id>", methods=["GET"])
def explain_recipe(recipe_id):
    user_id = request.args.get("user_id", "").strip()
    output_format = request.args.get("format", "html").strip().lower()

    if not user_id:
        flash("User ID is required for explanation.")
        return redirect(url_for("home"))

    if hybrid_engine is None:
        flash("Hybrid engine is not available.")
        return redirect(url_for("home"))

    try:
        # Pass svd_model explicitly
        explanation = hybrid_engine.explain(user_id=str(user_id), recipe_id=str(recipe_id), svd_model=svd_model, top_similar=5)

        # Force template rendering if format is html
        if output_format == "html":
            return render_template("explain.html", explanation=explanation)

        # Otherwise return JSON safely
        return jsonify(explanation)

    except Exception as e:
        logging.error(f"Explain failed: {e}")
        flash("Could not generate explanation.")
        return redirect(url_for("home"))


@app.route("/recipe_images/<filename>")
def serve_recipe_image(filename):
    """
    Serve recipe images from core-data-images or raw-data-images.
    """
    core_path = os.path.join(config.CORE_IMAGE_DIR, filename)
    if os.path.exists(core_path):
        return send_from_directory(config.CORE_IMAGE_DIR, filename)
    
    raw_path = os.path.join(config.RAW_IMAGE_DIR, filename)
    if os.path.exists(raw_path):
        return send_from_directory(config.RAW_IMAGE_DIR, filename)
    
    return "Image not found", 404


@app.route("/retrain_model", methods=["GET", "POST"])
def retrain_model_route():
    if request.method == "POST":
        with app.app_context():
            retrain_model()
        flash("Model retraining completed.")
        return redirect(url_for("home"))
    return render_template("retrain_model.html")


if __name__ == "__main__":
    with app.app_context():
        # Ensure database integrity (ingredients, instructions, image URLs)
        ensure_data_integrity()
        
        # Load or Build Hybrid Engine
        logging.info("Initializing Hybrid Recommendation Engine...")
        hybrid_engine = build_or_load_hybrid_engine()
        
    app.run(debug=True)
