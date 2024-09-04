from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\models\best_random_forest_model.pkl'
model = joblib.load(model_path)

# Define the feature names after SMOTE
feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51', 'x52', 'x54', 'x55', 
'x56', 'x57', 'x58', 'x59', 'x60', 'y.1', 'interaction_term_x43_x44', 'interaction_term_x45_x46', 'x43.1', 'x44.1', 'x45.1', 'x43^2', 'x43 x44', 'x43 x45', 'x44^2', 'x44 x45', 'x45^2', 
'log_x43', 'log_x44']

# Home route
@app.route('/')
def home():
    """Home page with a form to input data for prediction."""
    return render_template('index.html')

# Prediction route (API)
@app.route('/predict', methods=['POST'])
def predict():
    """API to make predictions using the trained model."""
    try:
        # Retrieve data from the form
        data = request.form.get('data')
        
        if data:
            # Convert the comma-separated string to a list of floats
            data_list = list(map(float, data.split(',')))
            
            # Check if the number of features matches the expected number (74 in this case)
            if len(data_list) != len(feature_names):
                return render_template('index.html', prediction_text="Error: Please provide 74 features.")
            
            # Convert the list to a DataFrame with the correct feature names
            data_df = pd.DataFrame([data_list], columns=feature_names)
            
            # Make prediction
            prediction = model.predict(data_df)[0]
            return render_template('index.html', prediction_text=f'Prediction: {int(prediction)}')

        else:
            return render_template('index.html', prediction_text="Please enter valid data.")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
