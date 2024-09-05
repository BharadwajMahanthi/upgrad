# Predictive Maintenance - Machine Breakdown Prediction

This project is aimed at predicting machine breakdown by identifying anomalies in the provided dataset. The dataset contains binary labels (1 for anomaly and 0 for normal) along with several predictive features.

## Project Overview

Predictive maintenance solutions are crucial for minimizing risks and taking preventive actions. This project builds a machine learning model to predict machine breakdown using a Random Forest Classifier. The main steps involved are:

1. **Data Preprocessing**: Handling missing values, outliers, and scaling features.
2. **Feature Engineering**: Creating interaction terms, polynomial features, and performing log transformations.
3. **Exploratory Data Analysis (EDA)**: Identifying important features, handling missing values, and checking correlations.
4. **Model Training**: Using a Random Forest Classifier with hyperparameter tuning to achieve optimal performance.
5. **Model Evaluation**: Accuracy, F1-score, precision, recall, ROC curve, and confusion matrix.
6. **Model Deployment**: API built with Flask to predict machine breakdown using the trained model.

## Repository Structure

- `data/`: Contains the raw and preprocessed datasets.
- `src/models/`: Contains the model training and prediction scripts.
- `src/features/`: Contains the feature engineering and preprocessing scripts.
- `src/visualization/`: Contains the scripts for visualizations.
- `models/`: Stores the saved models and evaluation results.
- `README.md`: This file, describing the project.

## Prerequisites

To run this project, you need:

- Python 3.9+
- Libraries:
  - `pandas`
  - `numpy`
  - `seaborn`
  - `matplotlib`
  - `sklearn`
  - `flask`
  - `joblib`

You can install the required libraries using:

```bash
pip install -r requirements.txt
