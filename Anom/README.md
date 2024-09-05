# Predictive Maintenance - Machine Breakdown Prediction

This project focuses on predicting machine breakdown by identifying anomalies within operational data. The dataset provided contains binary labels (`1` for anomaly and `0` for normal) alongside numerous predictive features. Predictive maintenance is critical for preventing system failures, and this project builds a machine learning model to assist with this task.

## Project Overview

Predictive maintenance solutions are essential for reducing downtime and preventing unexpected machine failures. This project leverages machine learning to identify potential breakdowns by detecting anomalies in the operational data. The primary components of the project are:

1. **Data Preprocessing**: Handles missing values, outliers, and scales features to prepare the dataset for model training.
2. **Feature Engineering**: Includes interaction terms, polynomial features, and log transformations to enhance model performance.
3. **Exploratory Data Analysis (EDA)**: Investigates the dataset to identify important features, handle missing values, and assess correlations.
4. **Model Training**: Trains a Random Forest Classifier using hyperparameter tuning (RandomizedSearchCV) to optimize model accuracy and performance.
5. **Model Evaluation**: Evaluates the model using accuracy, F1-score, precision, recall, ROC curve, and confusion matrix metrics.
6. **Model Deployment**: Develops a Flask API to serve predictions using the trained model.

## Repository Structure

```bash
├── Anom
│   ├── data                  # Contains raw and processed datasets
│   ├── models                # Contains trained models and evaluation results
│   ├── notebooks             # Jupyter notebooks for experiments and analysis
│   ├── reports               # Reports and model evaluations
│   ├── src                   # Source code for data loading, feature engineering, and model training
│   │   ├── data              # Data loading and preprocessing scripts
│   │   ├── features          # Feature engineering and transformation scripts
│   │   ├── models            # Model training and evaluation scripts
│   │   ├── app
│   │   ├── visualization
├── main.ipynb                # Main notebook for running the pipeline
├── random_forest_model.pkl    # Trained Random Forest model file
├── README.md                 # Project documentation

```

## Project Workflow

### 1. Data Preprocessing
- **Task**: Handle missing values, detect and remove outliers, and scale numerical features using standard scaling techniques.
- **Script**: `load_data.py` loads the dataset and performs necessary preprocessing steps to ensure the data is ready for model training.

### 2. Feature Engineering
- **Task**: Generate interaction terms, polynomial features, and apply transformations (e.g., log transformations) to boost model accuracy.
- **Script**: `build_features.py` implements these transformations and enhances the feature set for improved model performance.

### 3. Exploratory Data Analysis (EDA)
- **Task**: Conduct Univariate and Bivariate analysis, generate correlation heatmaps, and assess feature importance to gain insights into the data.
- **Script**: `load_data.py` also handles the EDA process and outputs visualizations and summary statistics.

### 4. Model Training
- **Task**: Train a Random Forest Classifier, and use RandomizedSearchCV for hyperparameter tuning to ensure the best possible performance.
- **Script**: `train_model.py` splits the data into train and test sets, trains the model, and performs hyperparameter optimization.

### 5. Model Evaluation
- **Task**: Evaluate the model using a variety of metrics (accuracy, F1-score, recall, precision). Visualizations such as confusion matrices, ROC curves, and feature importance plots are saved for analysis.
- **Script**: `train_model.py` generates these evaluations and saves them in both image and PDF format for review.

### 6. Model Deployment
- **Task**: Deploy the trained model using a Flask API, allowing real-time predictions by sending input features through a POST request.
- **Script**: `app.py` is the Flask-based API for real-time predictions.

## Prerequisites

Before running the project, ensure you have Python 3.9+ and the following libraries installed:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `flask`
- `joblib`

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/BharadwajMahanthi/upgrad.git
```

### 2. Navigate to the project directory
```bash
cd upgrad/capstone
```

### 3. Data Preprocessing and Feature Engineering
Run the scripts for loading and preprocessing data:
```bash
python src/data/load_data.py
python src/features/build_features.py
```

### 4. Model Training and Evaluation
Run the model training script to train and evaluate the Random Forest model:
```bash
python src/models/train_model.py
```

### 5. Deploying the Model
Start the Flask API for real-time predictions:
```bash
python src/app.py
```

Access the application by visiting `http://127.0.0.1:5000` in your browser.

## Model Performance and Results

- **Best Model**: Random Forest Classifier
- **Accuracy**: Achieved above 99% accuracy on the test dataset after hyperparameter tuning.
- **Metrics**: F1-score, recall, precision, ROC curve, and confusion matrix are used to evaluate model performance.
- **Overfitting**: No significant overfitting detected, based on train-test accuracy comparison.

## Future Work

1. **Model Enhancement**: Experimenting with more complex models (e.g., Gradient Boosting or Neural Networks) to improve performance further.
2. **Feature Engineering**: Applying additional feature engineering techniques, such as domain-specific transformations or feature selection.
3. **Model Explainability**: Implementing model explainability tools (e.g., SHAP values) to better understand model predictions.
4. **Continuous Learning**: Updating the model as more data is collected to ensure it stays relevant and accurate.
5. **Scaling Deployment**: Scaling the API using Docker or Kubernetes for better production readiness.

## Source Code

The source code for this project is structured in a modular fashion:

- `load_data.py`: Handles data loading, preprocessing, and initial EDA.
- `build_features.py`: Implements feature engineering.
- `train_model.py`: Contains the code for training and evaluating the model.
- `app.py`: Flask-based API for deploying the model and serving predictions.

You can view the source code files in the `src/` directory of the repository.

---

This documentation provides a detailed overview of the project structure, workflow, and future improvements. You can also add more sections based on additional features or insights gathered during the project.
