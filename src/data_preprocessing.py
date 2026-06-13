import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_check_raw_data(file_path):
    """
    Loads raw CSV data and performs data understanding analysis.
    Saves a report to outputs/reports/data_understanding_report.txt.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV dataset.
        
    Returns:
    --------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join("outputs", "reports"), exist_ok=True)
    report_path = os.path.join("outputs", "reports", "data_understanding_report.txt")
    
    # Perform Data Understanding
    num_rows, num_cols = df.shape
    columns_list = list(df.columns)
    data_types = df.dtypes
    first_5 = df.head()
    desc_stats = df.describe(include='all')
    missing_values = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    
    # Correlation analysis for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("          CRISP-DM Data Understanding Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"1. Dataset Shape:\n")
        f.write(f"   - Number of rows: {num_rows}\n")
        f.write(f"   - Number of columns: {num_cols}\n\n")
        f.write(f"2. Column Names:\n")
        f.write(f"   - {', '.join(columns_list)}\n\n")
        f.write("3. Data Types:\n")
        for col, dtype in data_types.items():
            f.write(f"   - {col}: {dtype}\n")
        f.write("\n")
        f.write("4. First 5 Rows:\n")
        f.write(first_5.to_string() + "\n\n")
        f.write("5. Descriptive Statistics:\n")
        f.write(desc_stats.to_string() + "\n\n")
        f.write("6. Missing Values Check:\n")
        for col, missing in missing_values.items():
            f.write(f"   - {col}: {missing} missing value(s)\n")
        f.write("\n")
        f.write("7. Duplicate Rows Check:\n")
        f.write(f"   - Number of duplicate rows: {duplicate_count}\n\n")
        f.write("8. Correlation Analysis (Numeric columns):\n")
        f.write(corr_matrix.to_string() + "\n\n")
        f.write("="*60 + "\n")
        
    print(f"Data understanding completed. Report saved to: {report_path}")
    return df

def preprocess_data(df):
    """
    Applies one-hot encoding to State column, avoiding dummy variable trap.
    Drops the 'California' column as the baseline.
    Renames encoded State columns to match required feature names.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input DataFrame.
        
    Returns:
    --------
    X : pd.DataFrame
        Preprocessed features.
    y : pd.Series
        Target variable (Profit).
    """
    # Verify columns exist
    required_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")
            
    # Perform one-hot encoding on State
    # Note: get_dummies will return California, Florida, New York
    state_dummies = pd.get_dummies(df['State'], dtype=int)
    
    # Avoid dummy variable trap: Drop 'California' as baseline
    if 'California' in state_dummies.columns:
        state_dummies = state_dummies.drop(columns=['California'])
        
    # Build feature DataFrame X
    # Required columns in feature set order (R&D Spend, Marketing Spend, New York, Florida, Administration)
    # Let's align features as required.
    X = pd.DataFrame({
        'R&D Spend': df['R&D Spend'],
        'Marketing Spend': df['Marketing Spend'],
        'New York': state_dummies['New York'] if 'New York' in state_dummies.columns else 0,
        'Florida': state_dummies['Florida'] if 'Florida' in state_dummies.columns else 0,
        'Administration': df['Administration']
    })
    
    y = df['Profit']
    
    return X, y

def split_data(X, y):
    """
    Splits the data into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : train/test splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
