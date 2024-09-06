# FindDefault Project

## Overview

The **FindDefault** project focuses on predicting loan defaults using **logistic regression**. The dataset contains detailed information about various loan applicants, including their financial status, credit history, and loan details. The primary goal of the project is to accurately predict the likelihood of a loan default based on these features.

## Project Structure

Here is the directory structure of the project:

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
│
```

### Key Directories and Files

- **data/raw**: Contains the raw dataset (`creditcard.csv`) used for training the logistic regression model.
- **data/processed**: Preprocessed data used for model training and evaluation.
- **models**: Contains the logistic regression models saved as `.pkl` files and a report (`logreg_model_results.pdf`) with results and insights from the logistic regression model.
- **notebooks**: Jupyter notebook (`main.ipynb`) that performs data preprocessing, feature engineering, and logistic regression model training.
- **src**: Contains Python scripts for loading data, feature engineering, training models, and visualizing results.

## Machine Learning Model Used

This project exclusively uses **logistic regression** for predicting loan defaults. Logistic regression was chosen for its simplicity and effectiveness in binary classification tasks. 

- **Logistic Regression Model**: The logistic regression model is used to classify loan applications as either likely to default or not.

## Data Preprocessing

Data preprocessing steps include:

- **Loading Data**: The raw data is loaded and cleaned using scripts in the `src/data/load_data.py` file.
- **Feature Engineering**: Features are built using `src/features/build_features.py` to improve model accuracy.
- **Scaling and Encoding**: Continuous variables are scaled, and categorical variables are encoded to prepare the data for modeling.

## Why EDA Techniques Are Not Applied

In this project, **Exploratory Data Analysis (EDA)** techniques are not extensively used for the following reasons:

1. **High Dimensionality of Data**: The dataset contains many features with complex relationships. While EDA is useful for visualizing patterns in small datasets, the dimensionality and complexity here limit the effectiveness of traditional EDA techniques.

2. **Privacy and Sensitivity of Data**: Since the data contains sensitive financial and personal information, minimizing manual inspection reduces the risk of inadvertent exposure.

3. **Focus on Automation**: The project is built with automation in mind, focusing on data pipelines and model training rather than manual data exploration.

4. **Model-Driven Insights**: The primary focus of the project is on generating insights through logistic regression models, rather than visual exploration of the data.

## Conclusion

This project leverages **logistic regression** to predict loan defaults, with a strong focus on automation and model-driven insights. By streamlining data preprocessing and focusing on machine learning, we can provide valuable predictions without the need for extensive manual data exploration.

---

Feel free to update this `README.md` as necessary. Let me know if any adjustments are required!
