import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, roc_curve, auc
from xgboost import XGBClassifier, DMatrix
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Ensure directories for saving plots and models exist
vis_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\src\visualization"
model_results_path_logreg = r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\models\logreg_model_results.pdf'
model_save_path_lgr = r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\models\best_lgr_model.pkl'
os.makedirs(vis_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_feature_engineered_data():
    """Load resampled and feature-engineered data (includes SMOTE)."""
    X = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\processed\X_features_resampled.csv')
    y = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\processed\y_resampled.csv')
    y = y.values.ravel()  # Ensure 'y' is flattened
    return X, y

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_metrics(accuracy, f1, recall, precision, save_path):
    """Plot and save accuracy, F1-score, recall, precision."""
    metrics = {'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'Precision': precision}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)
    plt.savefig(save_path)
    plt.show()
    plt.close()

def overfitting_check(train_accuracy, test_accuracy):
    """Check for overfitting by comparing training and test accuracy."""
    accuracy_gap = train_accuracy - test_accuracy
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Accuracy Gap (Overfitting check): {accuracy_gap:.4f}")
    if accuracy_gap > 0.05:
        print("Potential overfitting detected!")
        return "Potential overfitting detected!"
    else:
        print("No significant overfitting detected.")
        return "No significant overfitting detected."


def train_logistic_regression_torch(X, y, param_distributions=None, n_iter=5, num_epochs=100, lr=0.001, regularization=None, reg_lambda=0.01, model_save_path=None):
    """
    Train a logistic regression model with optional L1 or L2 regularization.
    
    Args:
        regularization: None, 'l1', or 'l2' for the type of regularization.
        reg_lambda: Regularization strength for L1 or L2.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert DataFrame to NumPy arrays if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # Convert to torch tensors and move to GPU (if available)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

    # Hyperparameter tuning setup (learning rate and num_epochs)
    if param_distributions is None:
        param_distributions = {
            'lr': [0.001, 0.01, 0.1],  # Learning rates to try
            'num_epochs': [50, 100, 200]  # Number of epochs to try
        }
    
    best_model = None
    best_f1 = 0
    best_params = None

    # Loop over random combinations of hyperparameters
    for i in range(n_iter):
        lr = np.random.choice(param_distributions['lr'])
        num_epochs = np.random.choice(param_distributions['num_epochs'])
        
        print(f"Training with learning rate: {lr}, epochs: {num_epochs}, regularization: {regularization}, lambda: {reg_lambda}")

        # Define the logistic regression model using PyTorch
        class LogisticRegressionTorch(nn.Module):
            def __init__(self, input_dim):
                super(LogisticRegressionTorch, self).__init__()
                self.linear = nn.Linear(input_dim, 1)
            
            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        # Model, loss function, and optimizer
        input_dim = X_train.shape[1]
        model = LogisticRegressionTorch(input_dim).to(device)

        # Use L2 regularization via `weight_decay` or manual L1 regularization
        if regularization == 'l2':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_lambda)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        criterion = nn.BCELoss()  # Binary cross entropy loss for classification

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train).squeeze()  # Flatten the output for BCE Loss
            loss = criterion(outputs, y_train)

            # Add L1 regularization manually
            if regularization == 'l1':
                l1_penalty = 0
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                loss += reg_lambda * l1_penalty

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test).squeeze()
            y_pred_proba = y_pred_test.clone()  # Save probabilities for ROC curve
            y_pred_test = torch.round(y_pred_test)  # Convert probabilities to 0 or 1
            
            train_accuracy = accuracy_score(y_train.cpu().numpy(), torch.round(model(X_train)).cpu().numpy())
            test_accuracy = accuracy_score(y_test.cpu().numpy(), y_pred_test.cpu().numpy())

            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            of=overfitting_check(train_accuracy, test_accuracy)
            print(of)
            
            # Convert tensors to NumPy for further processing
            y_test_np = y_test.cpu().numpy()
            y_pred_test_np = y_pred_test.cpu().numpy()
            y_pred_proba_np = y_pred_proba.cpu().numpy()

            # Evaluate Logistic Regression
            accuracy = accuracy_score(y_test_np, y_pred_test_np)
            f1 = f1_score(y_test_np, y_pred_test_np)
            recall = recall_score(y_test_np, y_pred_test_np)
            precision = precision_score(y_test_np, y_pred_test_np)
            # Create a DataFrame with both actual and predicted values
            comparison_df = pd.DataFrame({
                'Actual': y_pred_test_np,
                'Predicted': y_pred_proba_np
            })

            # Optionally, add a column to show if the prediction is correct
            comparison_df['Correct'] = comparison_df['Actual'] == comparison_df['Predicted']

            # Show the first few rows
            print(comparison_df.head())
            
            # Show rows where predictions were wrong
            wrong_predictions = comparison_df[comparison_df['Correct'] == False]
            print(wrong_predictions)


            print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")

            # If this model is better, update the best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_params = {'lr': lr, 'num_epochs': num_epochs, 'regularization': regularization, 'reg_lambda': reg_lambda}

    # After training with all hyperparameters, print the best result
    print(f"Best model with learning rate: {best_params['lr']}, epochs: {best_params['num_epochs']}, regularization: {best_params['regularization']}, lambda: {best_params['reg_lambda']}")
    print(f"Best F1 score: {best_f1}")

    # Save the best model
    if model_save_path is not None:
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Best model saved to {model_save_path}")
    

    # Final evaluation with the best model
    with torch.no_grad():
        y_pred_test = best_model(X_test).squeeze()
        y_pred_proba = y_pred_test.clone()  # Save probabilities for ROC curve
        y_pred_test = torch.round(y_pred_test)  # Convert probabilities to 0 or 1
        
        # Convert tensors to NumPy for further processing
        y_test_np = y_test.cpu().numpy()
        y_pred_test_np = y_pred_test.cpu().numpy()
        y_pred_proba_np = y_pred_proba.cpu().numpy()

        # Evaluate the best model
        accuracy = accuracy_score(y_test_np, y_pred_test_np)
        f1 = f1_score(y_test_np, y_pred_test_np)
        recall = recall_score(y_test_np, y_pred_test_np)
        precision = precision_score(y_test_np, y_pred_test_np)

        print(f"Final Best Logistic Regression Model Accuracy: {accuracy:.4f}")
        plot_confusion_matrix(y_test_np, y_pred_test_np, os.path.join(vis_path, 'logreg_best_confusion_matrix.png'))
        plot_roc_curve(y_test_np, y_pred_proba_np, os.path.join(vis_path, 'logreg_best_roc_curve.png'))
        plot_metrics(accuracy, f1, recall, precision, os.path.join(vis_path, 'logreg_best_metrics.png'))

    return y_pred_test_np  # Return the predicted values as a NumPy array


if __name__ == "__main__":
    # Load the feature-engineered data
    X, y = load_feature_engineered_data()


    param_distributions = {'lr': [0.001, 0.01, 0.1], 'num_epochs': [150, 300, 600]}
    regularization = 'l1'  # or 'l2' or None
    reg_lambda = 0.01

    y_pred_test = train_logistic_regression_torch(X, y, param_distributions=param_distributions, n_iter=5, regularization=regularization, reg_lambda=reg_lambda, model_save_path=model_save_path_lgr)


