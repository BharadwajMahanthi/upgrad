# upGrad Knowledgehut AI & Data Science Portfolio v2.0

![upGrad Logo](https://github.com/user-attachments/assets/4fb2d67f-fd69-4cc8-bb45-bce555143f74)

## 📖 Main Branch Projects

### Overview

This repository hosts multiple data science and machine learning projects, focusing on predictive modeling, anomaly detection, fire detection, recipe recommendation systems, and future projects in deep learning and AI. Each project aims to solve real-world problems using machine learning techniques, automation, and best practices in data science.

### Current Projects:

1. **FindDefault**: Predicting loan defaults using logistic regression.
2. **Anom**: High-precision anomaly detection with threshold optimization (**99.76% Accuracy**).
3. **Fire Detection (FS_SM_NSM)**: A deep learning-based system for detecting fire and smoke in images using CNN, MobileNetV2, and Grad-CAM.
4. **Nutrigo v2.0**: A comprehensive recipe recommendation system leveraging Flask and ML models, now enriched with **3.7M reviews**.
5. **Upcoming Projects from UpGrad**: More projects to be added from UpGrad courses and capstone projects.

---

## 1. **FindDefault Project**

### Project Overview

The **FindDefault** project focuses on predicting loan defaults using **logistic regression**. The dataset includes financial information about loan applicants, and the goal is to accurately predict which loans are likely to default based on these features.

### Project Structure

```plaintext
FindDefault/
│
├── data/
│   ├── processed/                        # Processed data used for modeling
│   └── raw/                              # Raw data, including creditcard.csv
│       └── creditcard.csv                # The raw dataset
│
├── models/
│   ├── best_lgr_model.pkl                # Best logistic regression model (pickled)
│   ├── best_logreg_model.pkl             # Another pickled logistic regression model
│   ├── logreg_model_results.pdf          # PDF with model results and analysis
│   └── test_app_model                    # Model used for the test application
│
├── notebooks/
│   └── main.ipynb                        # Jupyter notebook for data preprocessing and model training
│
├── src/
│   ├── app/
│   ├── data/
│   │   └── load_data.py                  # Script to load the data
│   ├── features/
│   │   └── build_features.py             # Script to build features for the model
│   ├── models/
│   │   └── train_model.py                # Script to train the logistic regression model
│   └── visualization/
```

### Key Points

- **Logistic Regression**: The project uses logistic regression to predict loan defaults.
- **Automated Preprocessing**: Data loading, feature engineering, and model training are automated.
- **Model Evaluation**: The model is evaluated based on accuracy, precision, recall, and F1-score.

### Why EDA Techniques Are Not Applied

- **High Dimensionality**: The dataset has many features, making traditional EDA less effective.
- **Privacy Considerations**: Reducing manual inspection due to the sensitive nature of the financial data.
- **Automation**: Focus is on building automated pipelines for feature engineering and model training.
- **Model-Driven Insights**: Insights are drawn from the logistic regression model itself, rather than visual data exploration.

---

## 2. **Anom Project**

### Project Overview

The **Anom** project is focused on detecting anomalies in a given dataset. This project involves using advanced machine learning models to classify anomalies, particularly in high-dimensional data, achieving a production-grade **99.76% accuracy**.

### Project Structure

```plaintext
Anom/
│
├── data/
│   ├── processed/                        # Processed data used for anomaly detection
│   └── raw/                              # Raw data used for training
│
├── models/
│   └── best_anom_model.pkl               # Best anomaly detection model (pickled)
│
├── notebooks/
│   └── anom_model_training.ipynb         # Jupyter notebook for anomaly detection model training
│
├── src/
│   ├── data/
│   │   └── load_data.py                  # Script to load anomaly detection data
│   ├── features/
│   │   └── build_features.py             # Script to build features for anomaly detection
│   └── models/
│       └── train_anom_model.py           # Script to train the anomaly detection model
```

### Key Points

- **Anomaly Detection**: Focuses on detecting anomalies using models like Isolation Forest, One-Class SVM, and `HistGradientBoostingClassifier`.
- **Threshold Optimization**: Automated decision boundary tuning (Optimal: **0.20**) for maximizing recall.
- **Preprocessing and Feature Engineering**: Automated data pipelines for preprocessing and interaction feature engineering.

---

## 3. **Fire Detection Project (FS_SM_NSM)**

### Project Overview

The **Fire Detection** project leverages deep learning to detect fire and smoke in real-time images using MobileNetV2. The dataset is sourced from Kaggle and includes images categorized as fire, smoke, and non-fire.

- **Dataset**: [Forest Fire, Smoke, and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)

### Project Structure

```plaintext
FireDetection/
│
├── keras_tuner/                          # Keras Tuner hyperparameter tuning outputs
├── output/                               # Directory for saved models and outputs
├── templates/                            # Templates for any web application views
│
├── app.py                                # Main Python file for Flask app (includes Grad-CAM)
├── Classification.ipynb                  # Jupyter notebook for fire detection model training
├── fire_detection.h5                     # Saved H5 model file
├── Readme.md                             # Project README file
```

### Key Points

- **Multi-Model Approach**: Uses CNN and MobileNetV2 architectures.
- **Hyperparameter Tuning**: Leverages Keras Tuner for optimizing model performance.
- **GPU Acceleration**: Implements CUDA and mixed precision for faster model training.
- **Visual Evidence (Grad-CAM)**: Real-time localization highlighting fire/smoke regions.

### GPU, CUDA, and NVIDIA Support

The project supports GPU acceleration using **CUDA** and **NVIDIA Tensor Cores** for fast training. The models are configured to use mixed precision for improved performance on compatible GPUs.

---

## 4. **Nutrigo Project v2.0**

### Project Overview

**Nutrigo** is a hyper-personalized recipe recommendation system designed to help users achieve healthy goals. Version 2.0 features a major upgrade in data scale and UI aesthetics.

### Project Structure

```plaintext
nutrigo/
│
├── app.py
├── classifier.ipynb
├── core-data-images/                # High-res local recipe images
├── core-data-test_rating.csv
├── core-data-train_rating.csv
├── core-data-valid_rating.csv
├── core-data_recipe.csv
├── data/
│   ├── combined_item_features.npz   # Managed by Git LFS
│   ├── core_image_features.npy      # Managed by Git LFS
│   └── raw_image_features.npy       # Managed by Git LFS
│   └── nutrigo.db                   # SQLite (WAL Mode)
├── models/
│   ├── healthiness_model.h5
│   ├── le_recipe.pkl                # Managed by Git LFS
│   ├── le_user.pkl                  # Managed by Git LFS
│   ├── scaler.pkl                   # Managed by Git LFS
│   ├── svd_model.pkl                # Managed by Git LFS
│   ├── tfidf_vectorizer.pkl         # Managed by Git LFS
│   └── training.log
├── static/
│   ├── css/
│   │   └── styles.css               # Modern Glassmorphism theme
│   └── images/
│       └── logo.png
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── update_profile.html
│   ├── recommendations.html
│   ├── recipe_detail.html
│   ├── admin_login.html
│   ├── admin_dashboard.html
│   └── retrain_model.html
├── migrations/                      # Flask-Migrate database evolution scripts
└── __pycache__/
```

### Key Points

- **Backend**: Flask, Flask-Migrate, Flask-Login, SQLAlchemy (WAL Mode).
- **Frontend**: HTML5, CSS3, Bootstrap 5 (Glassmorphism design).
- **Machine Learning**: TensorFlow (Health Audit), Scikit-learn (Content matching), Scikit-surprise (SVD).
- **Enrichment**: Integrated **3.7M reviews** and detailed macro/micronutrients.

### Features

- **User Authentication**: Secure registration and bio-profile calibration.
- **Hybrid Recommendations**: Tailored suggestions based on SVD and keyword matching.
- **AI Health Audit**: Real-time diagnostic audit with visualized probability scores.
- **Stability**: SQLite WAL mode ensures non-blocking concurrent data processing.

---

## 🚀 Global Installation & Usage

### 1. **Clone the Repository**

```bash
git clone https://github.com/BharadwajMahanthi/upgrad.git
cd upgrad
```

### 2. **Set Up Virtual Environment**

```bash
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 3. **Run Individual Projects**

- **Nutrigo**: `cd nutrigo && python app.py`
- **Fire Monitoring**: `cd FS_SM_NSM && python app.py`
- **Anomaly Detection**: `cd Anom && python src/app/app.py`

---

## 📁 Project Structure Overview

```plaintext
root/
├── Anom/                # High-precision anomaly detection
├── FindDefault/         # Credit Risk modeling
├── FS_SM_NSM/           # Fire detection with visual proof
├── nutrigo/             # AI Bio-Nutritionist v2.0
│   ├── app.py           # Main server
│   ├── data/            # Local SQLite (WAL) & CSV Snapshots
│   ├── static/          # Modern Assets
│   └── templates/       # Glassmorphism UI
├── output/              # Global artifacts & plots
├── .gitignore           # LFS and Exclusions (venv, .db, etc.)
└── requirements.txt     # Consolidated dependencies
```

### Key Governance:

- **Git LFS**: Large models (.h5, .pkl) and datasets are managed via Git Large File Storage.
- **Data Integrity**: Automated checks run on startup to ensures datasets are enriched and consistent.

---

## 🧰 Technologies Used

- **AI/ML**: TensorFlow, Keras, Scikit-learn, Matplotlib.
- **Backend**: Flask, Flask-Migrate, SQLAlchemy.
- **Data Engineering**: OpenCV, Pandas, NumPy, Jupyter Notebook.
- **Version Control**: Git, GitHub, Git LFS.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn and create.

### Steps to Contribute

1. **Fork the Project**.
2. **Clone Your Fork**.
3. **Create a New Branch**.
4. **Make Your Changes**.
5. **Commit Your Changes**.
6. **Push to the Branch**.
7. **Open a Pull Request**.

---

## 📜 License

Distributed under the **MIT License**.

---

## 📬 Contact

**Bharadwaj Mahanthi**

- **GitHub**: [@BharadwajMahanthi](https://github.com/BharadwajMahanthi)
- **Email**: mbpd.1999@gmail.com
- **Project Link**: [https://github.com/BharadwajMahanthi/upgrad](https://github.com/BharadwajMahanthi/upgrad)

---

## 📝 Acknowledgements

- upGrad Knowledgehut AI/ML Roadmaps.
- Flask, TensorFlow, and Scikit-Learn communities.
- NVIDIA CUDA Support for high-speed training.

---

_Developed by Bharadwaj Mahanthi_

---

## 🎉 Conclusion

Each project in this repository is meticulously organized to ensure clarity, ease of use, and scalability. Whether you're exploring **FindDefault, Anom, Fire Detection, or Nutrigo**, this repository serves as a comprehensive hub for elite data science endeavors.
