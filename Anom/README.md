# Predictive Maintenance & Anomaly Detection Pipeline

This project implements a **production-grade machine learning pipeline** for detecting machine anomalies. It leverages modern gradient boosting techniques, automated threshold optimization, and a robust data processing engine to identify rare failure events with high precision.

---

## 🏆 Key Achievements

- **Accuracy**: **99.76%**
- **F1-Score**: **0.8169** (Optimized from 0.76 baseline)
- **Precision**: **92.31%** (Extremely low false alarm rate)
- **Recall**: **64.86%** (Reliably catches critical anomalies)
- **Model**: `HistGradientBoostingClassifier` (Auto-selected over Random Forest & Logistic Regression via 5-Fold Stratified CV)

---

## 🚀 Features

1.  **Robust Data Pipeline**
    - Automatic cleaning (whitespace stripping, dropping leaked labels).
    - **Safe Feature Engineering**: `CustomFeatureEngineer` handles interactions, polynomials, and log transforms.
    - **Crash-Proof**: Automatically restores column metadata if raw numpy arrays are passed (e.g., from `SimpleImputer`).

2.  **Optimized Decision Making**
    - **Threshold Tuning**: System automatically finds the best decision threshold (Sensitivity vs. Precision) instead of using the default 0.5. Current optimal: **0.20**.
    - **Model Bundling**: The saved artifact (`best_model.pkl`) contains the _entire_ pipeline (preprocessing + model) AND the optimized threshold.

3.  **Deployment Ready**
    - **Flask API**: Professional REST API with error handling and logging.
    - **Web Dashboard**: Modern, responsive UI (`src/app/templates/index.html`) for real-time inference.
    - **Docker Ready**: Clean structure suitable for containerization.

---

## 📂 Repository Structure

```bash
Anom/
├── data/                   # Data storage
│   ├── raw/                # Original Excel files
│   └── processed/          # Cleaned intermediate data
├── models/                 # Artifacts
│   ├── best_model.pkl      # The PRODUCTION MODEL (Pipeline + Threshold)
│   └── model_results.csv   # Metric logs
├── src/                    # Source Code
│   ├── app/                # Flask Application
│   │   ├── templates/      # HTML UI
│   │   └── app.py          # API Server
│   ├── data/               # Loading & EDA
│   ├── features/           # Feature Engineering (CustomTransformers)
│   ├── models/             # Training & Evaluation Logic
│   └── visualization/      # Generated Plots (Confusion Matrix, ROC, etc.)
├── main.py                 # Master Training Pipeline Script
├── test_api_live.py        # Integration Test Script used for verification
├── requirements.txt        # Python Dependencies
└── README.md               # This file
```

---

## 🛠️ Installation & Usage

### 1. Setup Environment

Ensure Python 3.9+ is installed.

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the end-to-end pipeline. This will load data, perform EDA, engineer features, select the best model, tune the threshold, and save the artifact.

```bash
python main.py
```

_Artifacts will be saved to `models/` and `src/visualization/`._

### 3. Run the API Server

Start the Flask application for real-time inference.

```bash
python src/app/app.py
```

Access the dashboard at: **http://127.0.0.1:5000/**

### 4. Test the API

Verify the API is working using the provided test script (simulates Normal vs Anomaly requests).

```bash
python test_api_live.py
```

---

## 🔬 Exploratory Data Analysis (EDA)

The pipeline automatically generates insights in `src/visualization/`:

- **Correlation Heatmaps**: To identify redundant features.
- **Pairplots**: Visualizing relationships between top features.
- **KDE Plots**: Distribution differences between Normal (0) and Anomaly (1).
- **PCA Projection**: 2D scatter plot of the feature space.

---

## 🤝 Contribution

This pipeline is built to be modular.

- Modify `src/features/build_features.py` to add new transformations.
- Adjust `src/models/train_model.py` to add new algorithms to the competition.
- Update `src/config.py` to change paths or constants.

**Author**: Bharadwaj Mahanthi
