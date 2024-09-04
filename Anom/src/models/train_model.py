import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, roc_curve, auc
import joblib
import os
from matplotlib.backends.backend_pdf import PdfPages

# Define paths for saving the plots and model results
vis_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\anom\src\visualization"
model_results_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\models\model_results.pdf'
model_save_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\models\best_random_forest_model.pkl'

# Ensure the directories exist
os.makedirs(vis_path, exist_ok=True)

def load_feature_engineered_data():
    """Load resampled and feature-engineered data (includes SMOTE)."""
    X = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\processed\X_features_resampled.csv')
    y = pd.read_csv(r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\processed\y_resampled.csv')
    
    # Ensure 'y' is a 1D array (flatten it)
    y = y.values.ravel()
    
    return X, y

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)  # Save plot as PNG
    plt.show()
    plt.close()

def plot_feature_importance(model, X, save_path):
    """Plot and save feature importance of Random Forest model."""
    importances = model.feature_importances_
    indices = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=indices, y=indices.index)
    plt.title('Feature Importance')
    plt.savefig(save_path)  # Save plot as PNG
    plt.show()
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])  # Use the probabilities for the positive class
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
    plt.savefig(save_path)  # Save plot as PNG
    plt.show()
    plt.close()

def plot_metrics(accuracy, f1, recall, precision, save_path):
    """Plot and save accuracy, F1-score, recall, precision."""
    metrics = {'Accuracy': accuracy, 'F1 Score': f1, 'Recall': recall, 'Precision': precision}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)
    plt.savefig(save_path)  # Save plot as PNG
    plt.show()
    plt.close()

def train_model(X, y):
    """Train the Random Forest model on the CPU with RandomizedSearchCV."""
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize RandomForestClassifier with reasonable defaults
    rf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)
    
    # Define a smaller hyperparameter grid for RandomizedSearchCV
    param_distributions = {
        'n_estimators': [100, 200],  
        'max_depth': [10, 15, None],  
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],  
        'bootstrap': [True, False]
    }

    # Use RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=50, 
                                       cv=3, random_state=42, scoring='f1', verbose=2, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    # Get the best model from RandomizedSearchCV
    best_model = random_search.best_estimator_
    
    actual_features = X_train.columns.tolist()
    print(actual_features)

    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)  # For ROC curve
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # Overfitting check: Compare training and test accuracy
    train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
    accuracy_gap = train_accuracy - accuracy
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Accuracy Gap (Overfitting check): {accuracy_gap:.4f}")
    
    if accuracy_gap > 0.05:  
        print("Potential overfitting detected!")
    else:
        print("No significant overfitting detected.")
    
    # Save model results to PDF
    with PdfPages(model_results_path) as pdf:
        # Create a summary page with all key metrics and hyperparameters
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.01, 0.5, f"Model Accuracy: {accuracy:.4f}\nF1-Score: {f1:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n\nBest Hyperparameters:\n{random_search.best_params_}\n\nClassification Report:\n{classification_report(y_test, y_pred)}", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
    
    # Plot confusion matrix and save as PNG
    plot_confusion_matrix(y_test, y_pred, os.path.join(vis_path, 'confusion_matrix.png'))
    
    # Plot feature importance and save as PNG
    plot_feature_importance(best_model, X, os.path.join(vis_path, 'feature_importance.png'))
    
    # Plot ROC curve and save as PNG
    plot_roc_curve(y_test, y_pred_proba, os.path.join(vis_path, 'roc_curve.png'))
    
    # Plot model evaluation metrics and save as PNG
    plot_metrics(accuracy, f1, recall, precision, os.path.join(vis_path, 'model_metrics.png'))
    
    return best_model

if __name__ == "__main__":
    # Load the feature-engineered data
    X, y = load_feature_engineered_data()
    
    # Train the model on CPU
    model = train_model(X, y)
    
    # Save the trained model
    joblib.dump(model, model_save_path)
    print(f"Best model saved at {model_save_path}")
    print(f"Model results saved at {model_results_path}")
