from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import pandas as pd
from torch import nn

# Initialize the Flask app
app = Flask(__name__)

# Define the feature names
feature_names = ['PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 
                 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 
                 'PC20', 'PC21', 'PC22', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27', 'PC28']

# Logistic regression model class to match the saved state dict
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Load the trained model
model_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\models\best_logreg_model.pkl'

# Initialize the model and load the state dict
input_dim = len(feature_names)
model = LogisticRegressionTorch(input_dim)

# Set `weights_only=True` to safely load only the model weights
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()  # Set the model to evaluation mode

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
            
            # Check if the number of features matches the expected number (29 in this case)
            if len(data_list) != len(feature_names):
                return render_template('index.html', prediction_text="Error: Please provide 29 features.")
            
            # Convert the list to a NumPy array
            data_np = np.array([data_list], dtype=np.float32)

            # Convert to a torch tensor
            data_tensor = torch.tensor(data_np)

            # Make prediction using the model
            with torch.no_grad():
                prediction_proba = model(data_tensor).item()  # Get probability
                prediction = round(prediction_proba)  # Get class (0 or 1)

            return render_template('index.html', prediction_text=f'Prediction: {int(prediction)}, Probability: {prediction_proba:.4f}')

        else:
            return render_template('index.html', prediction_text="Please enter valid data.")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
