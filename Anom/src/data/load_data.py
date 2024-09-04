import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Define paths for saving the plots and results
output_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\reports'
vis_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\anom\src\visualization"
os.makedirs(output_path, exist_ok=True)

# Univariate analysis for continuous variables
def univariate_continuous(data, save_path):
    """Univariate analysis for continuous variables."""
    continuous_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in continuous_columns:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(data[col], kde=True, color='blue')
        plt.title(f'Histogram of {col}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[col], color='red')
        plt.title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'univariate_{col}.png'))
        plt.close()

# Univariate analysis for categorical variables
def univariate_categorical(data, save_path):
    """Univariate analysis for categorical variables."""
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data[col], palette='Set1')
        plt.title(f'Bar Plot of {col}')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(save_path, f'univariate_{col}.png'))
        plt.close()

# Bivariate analysis (continuous vs continuous)
def bivariate_continuous(data, save_path):
    """Bivariate analysis for continuous variables."""
    continuous_columns = data.select_dtypes(include=['float64', 'int64']).columns
    sns.pairplot(data[continuous_columns], diag_kind='kde')
    plt.title('Bivariate Analysis of Continuous Variables (Pairplot)')
    plt.savefig(os.path.join(save_path, 'bivariate_continuous_pairplot.png'))
    plt.close()

# Bivariate analysis (continuous vs categorical)
def bivariate_continuous_vs_categorical(data, save_path):
    """Bivariate analysis for continuous vs categorical variables."""
    continuous_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for cat_col in categorical_columns:
        for cont_col in continuous_columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=data[cat_col], y=data[cont_col], palette='Set3')
            plt.title(f'Boxplot of {cont_col} by {cat_col}')
            plt.xticks(rotation=90)
            plt.savefig(os.path.join(save_path, f'bivariate_{cont_col}_vs_{cat_col}.png'))
            plt.close()

# Correlation heatmap for continuous variables
def correlation_heatmap(data, save_path):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(save_path)
    plt.show()

# Main EDA function
def perform_eda(filepath):
    print("Perform EDA and preprocessing on the dataset.")
    # Load data
    data = pd.read_excel(filepath)

    # Univariate Analysis
    print("Performing Univariate Analysis...")
    univariate_continuous(data, vis_path)
    univariate_categorical(data, vis_path)

    # Bivariate Analysis
    print("Performing Bivariate Analysis...")
    bivariate_continuous(data, vis_path)
    bivariate_continuous_vs_categorical(data, vis_path)

    # Correlation Heatmap
    print("Generating Correlation Heatmap...")
    correlation_heatmap(data, os.path.join(vis_path, 'correlation_heatmap.png'))

    # Identify and save cleaned data for future steps
    data_cleaned = data.dropna()  # Example for further processing (can be replaced with imputation)
    processed_data_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\processed'
    os.makedirs(processed_data_path, exist_ok=True)

    data_cleaned.to_csv(os.path.join(processed_data_path, 'cleaned_data.csv'), index=False)
    print(f"Cleaned data saved to {processed_data_path}")

def plot_missing_values(data, save_path):
    """Plot and save missing values heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig(save_path)  # Save plot
    plt.show()
    
def preprocess_data(data):
    """Preprocess data by handling missing values, and scaling features."""
    
    # Drop any datetime columns
    datetime_cols = data.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        print(f"Dropping datetime columns: {list(datetime_cols)}")
        data = data.drop(columns=datetime_cols)
    
    # Define paths
    vis_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\anom\src\visualization"
    processed_data_path = r"C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\processed"
    
    # Create directories if they don't exist
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Plot and save missing values heatmap
    plot_missing_values(data, os.path.join(vis_path, 'missing_values_heatmap.png'))
    
    # Separate features (X) and target (y)
    X = data.drop('y', axis=1)
    y = data['y']
    
    # Handle missing values in X using mean imputation
    imputer_X = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)
    
    # Drop any remaining null values in X after imputation
    X_imputed.dropna(inplace=True)

    # Drop rows with missing `y` values
    y_cleaned = y.dropna()

    # Align X_imputed with y_cleaned (drop rows where y was NaN)
    X_aligned = X_imputed.loc[y_cleaned.index]
    
    # Check for outliers using box plots and save
    plt.figure(figsize=(12, 6))
    X_aligned.boxplot()
    plt.title('Box Plot for Outlier Detection')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(vis_path, 'outlier_boxplot.png'))  # Save plot
    plt.show()
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_aligned), columns=X_aligned.columns)
    
    # Concatenate X_scaled and y_cleaned into a single DataFrame
    p = pd.concat([X_scaled, y_cleaned.reset_index(drop=True)], axis=1)
    
    # Save preprocessed data with a new filename or overwrite existing file after checking
    preprocessed_data_file = os.path.join(processed_data_path, 'preprocessed_data.csv')
    
    if os.path.exists(preprocessed_data_file):
        print(f"File '{preprocessed_data_file}' already exists. Overwriting...")
    
    p.to_csv(preprocessed_data_file, index=False)
    print(f"Preprocessed data saved to {preprocessed_data_file}")
    
    # Save preprocessed data
    X_scaled.to_csv(os.path.join(processed_data_path, 'X_scaled.csv'), index=False)
    y_cleaned.to_csv(os.path.join(processed_data_path, 'y.csv'), index=False)
    
    return X_scaled, y_cleaned, p

if __name__ == "__main__":
    # Filepath to dataset
    data_path = r'C:\Users\mbpd1\downloads\upgrad\capstone\anom\data\raw\AnomaData.xlsx'
    perform_eda(data_path)
    
    # Preprocess the data
    X_scaled, y, preprocessed_data = preprocess_data(data_path)

