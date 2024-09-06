import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import cupy as cp  # cuPy for GPU-accelerated array processing
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths for saving the plots and results
output_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\reports'
vis_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\src\visualization"
processed_data_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\processed"
os.makedirs(output_path, exist_ok=True)

# Data Overview and Insights
def data_overview(data):
    """Display basic data information such as data types, missing values, and statistical summary."""
    # Convert cuPy array to pandas DataFrame
    data_df = pd.DataFrame(cp.asnumpy(data))

    print("\n--- Dataset Overview ---")
    print("Shape of the dataset:", data_df.shape)
    
    print("\n--- Data Types ---")
    print(data_df.dtypes)
    
    print("\n--- Checking for Missing Values ---")
    print(data_df.isnull().sum())
    
    print("\n--- Statistical Summary ---")
    print(data_df.describe())

# Univariate analysis for continuous variables
def univariate_continuous(data, save_path):
    """Univariate analysis for continuous variables."""
    # Convert cuPy array back to pandas DataFrame
    data_df = pd.DataFrame(cp.asnumpy(data))

    # Identify continuous columns (float64, int64)
    continuous_columns = data_df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in continuous_columns:
        plt.figure(figsize=(10, 5))
        
        # Convert cuPy array to numpy for plotting
        plt.subplot(1, 2, 1)
        sns.histplot(cp.asnumpy(data[:, col]), kde=True, color='blue')  # Use cuPy for GPU processing
        plt.title(f'Histogram of {col}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=cp.asnumpy(data[:, col]), color='red')  # Use cuPy for GPU processing
        plt.title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'univariate_{col}.png'))
        plt.show()
        plt.close()

# Bivariate analysis (continuous vs continuous)
def bivariate_continuous(data, save_path):
    """Bivariate analysis for continuous variables."""
    # Convert cuPy array back to pandas DataFrame for seaborn pairplot
    data_df = pd.DataFrame(cp.asnumpy(data))

    continuous_columns = data_df.select_dtypes(include=['float64', 'int64']).columns
    pairplot_data = data_df[continuous_columns]
    sns.pairplot(pairplot_data, diag_kind='kde')
    plt.title('Bivariate Analysis of Continuous Variables (Pairplot)')
    plt.savefig(os.path.join(save_path, 'bivariate_continuous_pairplot.png'))
    plt.show()
    plt.close()

# Correlation heatmap for continuous variables
def correlation_heatmap(data, save_path):
    """Plot correlation heatmap."""
    # Calculate correlation matrix on GPU, then convert to numpy for plotting
    plt.figure(figsize=(14, 10))  # Increase figure size
    corr = cp.asnumpy(cp.corrcoef(data.T))  # GPU-accelerated correlation matrix
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8}, linewidths=.5)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
    plt.yticks(fontsize=10)  # Set y-axis label size
    plt.title('Correlation Heatmap', fontsize=15)
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.savefig(save_path)
    plt.show()

# Main EDA function
def perform_eda(filepath):
    print("Performing EDA on the dataset.")
    try:
        # Load data
        data = pd.read_csv(filepath)

        # Convert to cuPy for GPU processing
        data_cp = cp.asarray(data)

        # Data Overview
        data_overview(data_cp)

        # Univariate Analysis
        print("Performing Univariate Analysis...")
        #univariate_continuous(data_cp, vis_path)

        # Bivariate Analysis
        print("Performing Bivariate Analysis...")
        #bivariate_continuous(data_cp, vis_path)

        # Correlation Heatmap
        print("Generating Correlation Heatmap...")
        correlation_heatmap(data_cp, os.path.join(vis_path, 'correlation_heatmap.png'))

        # Identify and save cleaned data for future steps
        data_cleaned = data.dropna()  # Example for further processing (can be replaced with imputation)

        data_cleaned.to_csv(os.path.join(processed_data_path, 'cleaned_data.csv'), index=False)
        print(f"Cleaned data saved to {processed_data_path}")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")

def plot_missing_values(data, save_path):
    """Plot and save missing values heatmap."""
    # Convert cuPy array to pandas DataFrame
    data_df = pd.DataFrame(cp.asnumpy(data))

    plt.figure(figsize=(10, 6))
    sns.heatmap(data_df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig(save_path)  # Save plot
    plt.show()
    
def preprocess_data(filepath):
    """Preprocess data by handling missing values and scaling features on GPU using cuPy and PyTorch."""
    # Load data
    data = pd.read_csv(filepath)
    
    # Drop any datetime columns (not present in this dataset)
    datetime_cols = data.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        print(f"Dropping datetime columns: {list(datetime_cols)}")
        data = data.drop(columns=datetime_cols)
    
    # Create directories if they don't exist
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Plot and save missing values heatmap
    plot_missing_values(data, os.path.join(vis_path, 'missing_values_heatmap.png'))
    
    # Separate features (X) and target (y)
    if 'Class' not in data.columns:
        raise ValueError("Target column 'Class' not found in dataset")
        
    X = data.drop('Class', axis=1)  # Assuming the target column is 'Class'
    y = data['Class']

    # Align X and y **before** processing X
    X, y = X.align(y, join='inner', axis=0)

    # Ensure y_cleaned is flattened (1-dimensional)
    y_cleaned = y.values.ravel()

    # Check class distribution before proceeding
    print("Class distribution before imputing missing values and scaling:")
    print(pd.Series(y_cleaned).value_counts())

    # Handle missing values in X using cuPy for mean imputation
    X_cp = cp.asarray(X)
    means_cp = cp.mean(X_cp, axis=0)  # Calculate the mean for each column

    # Iterate through columns and replace NaN with the respective column mean
    for i in range(X_cp.shape[1]):
        X_cp[:, i] = cp.nan_to_num(X_cp[:, i], nan=means_cp[i])

    # Now `X_cp` is imputed
    X_imputed_cp = X_cp

    # Check class distribution after alignment
    print("Class distribution after aligning X and y:")
    print(pd.Series(y_cleaned).value_counts())
    
    # Move data to GPU using cuPy and PyTorch tensors
    X_imputed_tensor = torch.tensor(X_imputed_cp.get(), dtype=torch.float32).to(device)
    y_cleaned_tensor = torch.tensor(y_cleaned, dtype=torch.float32).to(device)
    
    # Scaling numerical features using cuPy StandardScaler
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_imputed_tensor.cpu().numpy())  # Scale on CPU, move to GPU if needed
    X_scaled_tensor = torch.tensor(X_scaled_np, dtype=torch.float32).to(device)
    
    # Check for outliers using box plots and save
    plt.figure(figsize=(12, 6))
    pd.DataFrame(X_scaled_np, columns=X.columns).boxplot()
    plt.title('Box Plot for Outlier Detection')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(vis_path, 'outlier_boxplot.png'))  # Save plot
    plt.show()
    
    # Convert back to pandas for saving the data
    X_scaled = pd.DataFrame(X_scaled_tensor.cpu().numpy(), columns=X.columns)
    y_cleaned = pd.Series(y_cleaned_tensor.cpu().numpy(), name='Class')
    
    # Concatenate X_scaled and y_cleaned into a single DataFrame
    p = pd.concat([X_scaled, y_cleaned.reset_index(drop=True)], axis=1)
    
    # Save preprocessed data with a new filename or overwrite existing file after checking
    preprocessed_data_file = os.path.join(processed_data_path, 'preprocessed_data.csv')
    
    if os.path.exists(preprocessed_data_file):
        print(f"File '{preprocessed_data_file}' already exists. Overwriting...")
    
    p.to_csv(preprocessed_data_file, index=False)
    print(f"Preprocessed data saved to {preprocessed_data_file}")
    
    # Save X_scaled and y_cleaned separately
    X_scaled.to_csv(os.path.join(processed_data_path, 'X_scaled.csv'), index=False)
    y_cleaned.to_csv(os.path.join(processed_data_path, 'y.csv'), index=False)
    
    return X_scaled, y_cleaned, p


if __name__ == "__main__":
    # Filepath to dataset
    data_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\raw\creditcard.csv'
    data_path_cl = r'C:\Users\mbpd1\downloads\upgrad\capstone\FindDefault\data\processed\cleaned_data.csv'
    
    # Perform EDA on the raw dataset
    perform_eda(data_path)
    
    # Preprocess the cleaned data
    X_scaled, y, preprocessed_data = preprocess_data(data_path_cl)
