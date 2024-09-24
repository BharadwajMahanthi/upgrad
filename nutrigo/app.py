from flask import Flask, render_template, request, redirect, url_for, flash, session
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

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nutrigo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Global variables for the model and label encoders
svd_model = None
le_user = None
le_recipe = None

# Define paths for data files
CORE_RECIPE_PATH = 'core-data_recipe.csv'
RAW_RECIPE_PATH = 'raw-data_recipe.csv'
CORE_TRAIN_PATH = 'core-data-train_rating.csv'
CORE_VALID_PATH = 'core-data-valid_rating.csv'
CORE_TEST_PATH = 'core-data-test_rating.csv'
RAW_INTERACTION_PATH = 'raw-data_interaction.csv'

CORE_IMAGE_DIR = 'core-data-images'
RAW_IMAGE_DIR = 'raw-data-images'

# Define paths for saved features and models
CORE_IMAGE_FEATURES_PATH = 'core_image_features.npy'
RAW_IMAGE_FEATURES_PATH = 'raw_image_features.npy'
TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
SCALER_PATH = 'scaler.pkl'
HEALTH_MODEL_PATH = 'healthiness_model.h5'
LE_USER_PATH = 'le_user.pkl'
LE_RECIPE_PATH = 'le_recipe.pkl'
SVD_MODEL_PATH = 'svd_model.pkl'

# Define nutritional columns
TARGET_NUTRIENTS = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber']

# Function to parse the 'nutritions' column
def parse_nutritions_nested(df):
    # (Function code remains the same)
    logging.info("Parsing 'nutritions' column with nested dictionary structure...")

    # Function to clean the nutrition strings
    def clean_nutrition_str(nutrition_str):
        if isinstance(nutrition_str, str):
            # Replace single quotes with double quotes for JSON compatibility
            cleaned_str = nutrition_str.replace("u'", "'")
            return cleaned_str
        return '{}'

    # Clean the nutrition strings
    cleaned_nutritions = df['nutritions'].apply(clean_nutrition_str)

    # Safely parse using ast.literal_eval with exception handling
    def safe_literal_eval(nutrition_str):
        try:
            return ast.literal_eval(nutrition_str)
        except Exception as e:
            logging.error(f"Error parsing nutritions: {e} for string: {nutrition_str}")
            return {}

    nutritions_expanded = cleaned_nutritions.apply(safe_literal_eval)

    # Extract only the target nutrients
    def extract_nutrients(nutrition_dict):
        nutrient_values = {}
        for nutrient in TARGET_NUTRIENTS:
            if nutrient in nutrition_dict:
                nutrition_info = nutrition_dict[nutrient]
                if isinstance(nutrition_info, dict):
                    amount = nutrition_info.get('amount', np.nan)
                    if isinstance(amount, str):
                        amount = ''.join(filter(str.isdigit, amount))
                    nutrient_values[nutrient] = float(amount) if amount else np.nan
                elif isinstance(nutrition_info, (int, float)):
                    nutrient_values[nutrient] = nutrition_info
                else:
                    logging.warning(f"Unexpected format for nutrient '{nutrient}'. Assigning NaN.")
                    nutrient_values[nutrient] = np.nan
            else:
                logging.warning(f"Nutrient '{nutrient}' not found in nutrition data. Assigning NaN.")
                nutrient_values[nutrient] = np.nan
        return pd.Series(nutrient_values)

    nutritions_df = nutritions_expanded.apply(extract_nutrients)

    # Handle missing values by filling NaNs with the mean of each nutrient
    nutritions_df = nutritions_df.fillna(nutritions_df.mean())

    # Merge the extracted nutrients back into the original DataFrame
    df = pd.concat([df, nutritions_df], axis=1)

    # Drop the original 'nutritions' column
    df.drop('nutritions', axis=1, inplace=True)

    # Ensure that nutrient columns are of float type
    for col in TARGET_NUTRIENTS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill any remaining NaNs with the mean
    df[TARGET_NUTRIENTS] = df[TARGET_NUTRIENTS].fillna(df[TARGET_NUTRIENTS].mean())

    logging.info("Parsing 'nutritions' column completed.")
    return df

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), unique=True, nullable=False)
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    height = db.Column(db.Float)
    goal = db.Column(db.String(100))  # e.g., 'diet', 'muscle growth'
    health_details = db.Column(db.Text)
    preferences = db.Column(db.Text)  # e.g., 'vegan, gluten-free'
    interactions = relationship('Interaction', back_populates='user')

class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.String(50), unique=True, nullable=False)
    recipe_name = db.Column(db.String(255))
    ingredients = db.Column(db.Text)
    instructions = db.Column(db.Text)  # Add this line
    image_url = db.Column(db.String(255))
    calories = db.Column(db.Float)
    protein = db.Column(db.Float)
    fat = db.Column(db.Float)
    carbohydrates = db.Column(db.Float)
    fiber = db.Column(db.Float)
    interactions = relationship('Interaction', back_populates='recipe')
    reviews = relationship('Review', back_populates='recipe')  # Add this line for reviews

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), db.ForeignKey('user.user_id'))
    recipe_id = db.Column(db.String(50), db.ForeignKey('recipe.recipe_id'))
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
    user = relationship('User', back_populates='interactions')
    recipe = relationship('Recipe', back_populates='interactions')

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.String(50), db.ForeignKey('recipe.recipe_id'))
    user_id = db.Column(db.String(50))
    rating = db.Column(db.Float)
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime)
    recipe = relationship('Recipe', back_populates='reviews')

# Admin credentials (for simplicity)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD_HASH = generate_password_hash('admin123')  # Replace 'admin123' with your desired password

# Function to populate the database from CSV files
def populate_database():
    # Check if the database is empty
    recipe_count = Recipe.query.count()
    interaction_count = Interaction.query.count()

    if recipe_count == 0:
        logging.info("Populating database from CSV files...")

        # Load recipes data
        if os.path.exists(CORE_RECIPE_PATH):
            recipes_df = pd.read_csv(CORE_RECIPE_PATH)
            recipes_df = parse_nutritions_nested(recipes_df)
            recipes_df['recipe_id'] = recipes_df['recipe_id'].astype(str)

            # Remove 'instructions' and 'rating' columns if they don't exist
            expected_columns = ['recipe_id', 'recipe_name', 'ingredients', 'instructions', 'image_url',
                                'calories', 'protein', 'fat', 'carbohydrates', 'fiber']
            missing_columns = set(expected_columns) - set(recipes_df.columns)
            if missing_columns:
                logging.warning(f"Missing columns in recipes CSV: {missing_columns}")
                for col in missing_columns:
                    recipes_df[col] = None  # Assign None to missing columns

            for _, row in recipes_df.iterrows():
                recipe = Recipe(
                    recipe_id=row['recipe_id'],
                    recipe_name=row['recipe_name'],
                    ingredients=row['ingredients'],
                    instructions=row['instructions'],  # Include instructions
                    image_url=row['image_url'],
                    calories=row['calories'],
                    protein=row['protein'],
                    fat=row['fat'],
                    carbohydrates=row['carbohydrates'],
                    fiber=row['fiber']
                )
                db.session.add(recipe)

            db.session.commit()
            logging.info("Recipes data populated.")
        else:
            logging.error(f"Recipe CSV file '{CORE_RECIPE_PATH}' not found.")
            return

    if interaction_count == 0:
        # Load interactions data
        interaction_file_path = CORE_TRAIN_PATH  # Use the appropriate interactions file
        if os.path.exists(interaction_file_path):
            logging.info("Populating interactions from CSV file...")
            interactions_df = pd.read_csv(interaction_file_path)
            interactions_df['user_id'] = interactions_df['user_id'].astype(str)
            interactions_df['recipe_id'] = interactions_df['recipe_id'].astype(str)

            # Add users from interactions
            user_ids = interactions_df['user_id'].unique()
            for user_id in user_ids:
                existing_user = User.query.filter_by(user_id=user_id).first()
                if not existing_user:
                    user = User(user_id=user_id)
                    db.session.add(user)

            for _, row in interactions_df.iterrows():
                interaction = Interaction(
                    user_id=row['user_id'],
                    recipe_id=row['recipe_id'],
                    rating=row['rating'],
                    timestamp=row.get('timestamp')
                )
                db.session.add(interaction)

            db.session.commit()
            logging.info("Interactions data populated.")
        else:
            logging.warning(f"Interactions CSV file '{interaction_file_path}' not found. Skipping interactions data population.")
    else:
        logging.info("Database already populated.")

# Function to retrain the model
def retrain_model():
    # (Function code remains the same)
    global svd_model, le_user, le_recipe

    logging.info("Retraining the recommendation model...")

    # Load interaction data from the database
    interactions = Interaction.query.all()
    if not interactions:
        logging.warning("No interaction data available for model training.")
        svd_model = None
        return

    data = []
    for interaction in interactions:
        data.append((interaction.user_id, interaction.recipe_id, interaction.rating))

    df = pd.DataFrame(data, columns=['user_id', 'recipe_id', 'rating'])

    # Create label encoders for users and recipes
    from sklearn.preprocessing import LabelEncoder
    le_user = LabelEncoder()
    le_recipe = LabelEncoder()

    df['user_id_enc'] = le_user.fit_transform(df['user_id'])
    df['recipe_id_enc'] = le_recipe.fit_transform(df['recipe_id'])

    # Save the label encoders
    joblib.dump(le_user, LE_USER_PATH)
    joblib.dump(le_recipe, LE_RECIPE_PATH)

    # Prepare data for Surprise
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)

    # Build the full training set
    trainset = data.build_full_trainset()

    # Initialize and train the SVD model
    svd_model = SVD()
    svd_model.fit(trainset)

    # Save the trained model
    joblib.dump(svd_model, SVD_MODEL_PATH)

    logging.info("Model retraining completed.")

# Initialization function
def initialize_app():
    global svd_model, le_user, le_recipe

    # Check if database file exists
    if not os.path.exists('nutrigo.db'):
        logging.info("Database file 'nutrigo.db' not found. Creating a new database.")
        with app.app_context():
            db.create_all()
            populate_database()
            retrain_model()
    else:
        logging.info("Database file 'nutrigo.db' already exists. Skipping database creation and population.")
        # Load the model and label encoders
        with app.app_context():
            if os.path.exists(SVD_MODEL_PATH) and os.path.exists(LE_USER_PATH) and os.path.exists(LE_RECIPE_PATH):
                svd_model = joblib.load(SVD_MODEL_PATH)
                le_user = joblib.load(LE_USER_PATH)
                le_recipe = joblib.load(LE_RECIPE_PATH)
                logging.info("Model and label encoders loaded successfully.")
            else:
                logging.info("Model or label encoder files not found. Retraining model.")
                retrain_model()

# Call the initialization function
initialize_app()

# Helper function to check admin login
def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin login route
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['admin_logged_in'] = True
            flash('Logged in as admin.')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.')
            return redirect(url_for('admin_login'))
    return render_template('admin_login.html')

# Admin logout route
@app.route('/admin/logout')
@admin_required
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully.')
    return redirect(url_for('home'))

# Admin dashboard
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)

# Admin route to delete a user
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if user:
        # Delete user's interactions
        Interaction.query.filter_by(user_id=user.user_id).delete()
        # Delete user's reviews
        Review.query.filter_by(user_id=user.user_id).delete()
        # Delete the user
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.user_id} deleted successfully.')
    else:
        flash('User not found.')
    return redirect(url_for('admin_dashboard'))

# Route to display recipe details
@app.route('/recipe/<recipe_id>', methods=['GET', 'POST'])
def recipe_detail(recipe_id):
    # Fetch recipe from the database
    recipe = Recipe.query.filter_by(recipe_id=recipe_id).first()
    if not recipe:
        flash('Recipe not found.')
        return redirect(url_for('home'))

    # Handle review submission
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        rating = request.form.get('rating')
        comment = request.form.get('comment')

        review = Review(
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            rating=float(rating),
            comment=comment,
            timestamp=datetime.utcnow()
        )
        db.session.add(review)
        db.session.commit()
        flash('Your review has been submitted.')

    # Fetch reviews for the recipe
    reviews = Review.query.filter_by(recipe_id=recipe_id).order_by(Review.timestamp.desc()).all()

    return render_template('recipe_detail.html', recipe=recipe, reviews=reviews)

# Function to get top N recommendations
def get_recommendations(user_id, preferences_input=None, n_recommendations=10):
    # Fetch user from database
    user = User.query.filter_by(user_id=user_id).first()
    if not user:
        logging.warning(f"User ID {user_id} not found in database.")
        return []

    preferences = user.preferences
    goal = user.goal

    # Combine user preferences with input preferences
    if preferences_input:
        if preferences:
            preferences += f", {preferences_input}"
        else:
            preferences = preferences_input

    # Retrieve all recipes from the database
    recipes = Recipe.query.all()
    recipes_df = pd.DataFrame([{
        'recipe_id': recipe.recipe_id,
        'recipe_name': recipe.recipe_name,
        'ingredients': recipe.ingredients,
        'image_url': recipe.image_url,
        'calories': recipe.calories,
        'protein': recipe.protein
    } for recipe in recipes])

    if svd_model and le_user and le_recipe and user_id in le_user.classes_:
        # Existing user with interaction data
        user_index = le_user.transform([user_id])[0]

        # Get all recipe IDs and indices
        all_recipe_ids = recipes_df['recipe_id'].tolist()
        all_recipe_indices = le_recipe.transform(all_recipe_ids)

        # Predict ratings for all recipes
        predictions = []
        for recipe_id, recipe_index in zip(all_recipe_ids, all_recipe_indices):
            pred = svd_model.predict(user_index, recipe_index)
            predictions.append((recipe_id, pred.est))

        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions, columns=['recipe_id', 'est_rating'])

        # Merge with recipes DataFrame to get recipe details
        recommendations_df = pd.merge(pred_df, recipes_df, on='recipe_id')
    else:
        # New user or no trained model
        # Provide default recommendations (e.g., popular recipes)
        recommendations_df = recipes_df.copy()
        recommendations_df['est_rating'] = 0.0  # Default estimated rating

    # Filter based on preferences if provided
    if preferences:
        preference_list = [pref.strip().lower() for pref in preferences.split(',')]
        def match_preferences(ingredients):
            if ingredients:
                ingredients = ingredients.lower()
                return all(pref in ingredients for pref in preference_list)
            else:
                return False
        recommendations_df = recommendations_df[recommendations_df['ingredients'].apply(match_preferences)]

    # Filter based on user's goal (e.g., low calorie for diet)
    if goal:
        if 'diet' in goal.lower():
            # For diet goals, prefer low-calorie recipes
            recommendations_df = recommendations_df.sort_values(
                by=['calories', 'est_rating'], ascending=[True, False])
        elif 'muscle' in goal.lower():
            # For muscle growth, prefer high-protein recipes
            recommendations_df = recommendations_df.sort_values(
                by=['protein', 'est_rating'], ascending=[False, False])
        else:
            # Default sorting by estimated rating
            recommendations_df = recommendations_df.sort_values(by='est_rating', ascending=False)
    else:
        recommendations_df = recommendations_df.sort_values(by='est_rating', ascending=False)

    # Get top N recommendations
    top_recommendations = recommendations_df.head(n_recommendations)

    # Prepare the list of recommendations
    recommendations = []
    for _, row in top_recommendations.iterrows():
        calories = row.get('calories', 'N/A')
        if pd.isnull(calories):
            calories = 'N/A'
        else:
            calories = int(calories)
        protein = row.get('protein', 'N/A')
        if pd.isnull(protein):
            protein = 'N/A'
        else:
            protein = int(protein)
        image_url = row.get('image_url', '')
        if pd.isnull(image_url) or image_url.strip() == '':
            image_url = None  # Use None to indicate missing image
        recommendations.append({
            'recipe_id': row['recipe_id'],  # Include recipe_id
            'recipe_name': row['recipe_name'],
            'calories': calories,
            'image_url': image_url,
            'est_rating': round(row.get('est_rating', 0), 2),
            'protein': protein
        })

    return recommendations

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # (Route code remains the same)
    if request.method == 'POST':
        # Get user input
        user_id = request.form.get('user_id').strip()
        # Check if user_id already exists
        existing_user = User.query.filter_by(user_id=user_id).first()
        if existing_user:
            flash('User ID already exists. Please choose a different one.')
            return redirect(url_for('register'))
        # Collect other fields
        age = request.form.get('age')
        weight = request.form.get('weight')
        height = request.form.get('height')
        goal = request.form.get('goal')
        health_details = request.form.get('health_details')
        preferences = request.form.get('preferences')

        # Create new user
        new_user = User(
            user_id=user_id,
            age=int(age) if age else None,
            weight=float(weight) if weight else None,
            height=float(height) if height else None,
            goal=goal,
            health_details=health_details,
            preferences=preferences
        )

        # Save to database
        db.session.add(new_user)
        db.session.commit()

        flash(f'Registration successful! Your User ID is {user_id}. Please use it to log in.')
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    # (Route code remains the same)
    if request.method == 'POST':
        user_id = request.form.get('user_id').strip()
        user = User.query.filter_by(user_id=user_id).first()
        if not user:
            flash('User ID not found.')
            return redirect(url_for('update_profile'))

        # Update user information
        age = request.form.get('age')
        weight = request.form.get('weight')
        height = request.form.get('height')
        goal = request.form.get('goal')
        health_details = request.form.get('health_details')
        preferences = request.form.get('preferences')

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
        flash('Profile updated successfully.')
        return redirect(url_for('home'))

    return render_template('update_profile.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # (Route code remains the same)
    user_id = request.form.get('user_id').strip()
    preferences_input = request.form.get('preferences', '').strip()
    logging.info(f"Received recommendation request for User ID: {user_id}")

    # Get recommendations
    recommendations = get_recommendations(user_id, preferences_input)

    if not recommendations:
        error_message = "No recommendations found. Please check your User ID or register if you are a new user."
        return render_template('recommendations.html', error_message=error_message)

    # Render the recommendations template
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/retrain_model', methods=['GET', 'POST'])
def retrain_model_route():
    # (Route code remains the same)
    if request.method == 'POST':
        # Retrain the model
        with app.app_context():
            retrain_model()
        flash('Model retraining initiated.')
        return redirect(url_for('home'))
    return render_template('retrain_model.html')

if __name__ == '__main__':
    app.run(debug=True)
